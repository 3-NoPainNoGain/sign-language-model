from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json, time, unicodedata
import numpy as np
import mediapipe as mp
import cv2
from collections import Counter, deque
from typing import Any, List, Tuple

from ai_model.predict import load_model, predict_from_keypoints
from ai_model.preprocessing import decode_base64_image, extract_keypoints

app = FastAPI()

# ====== 파라미터 ======
WIN            = 6     # 최근 예측 창 크기
MAJ            = 4     # 다수결 임계
INACTIVITY_SEC = 1.2   # 마지막 확정단어 이후 입력 뜸하면 강제 플러시
HARD_RESET_SEC = 5.0   # 하드 리셋
COOLDOWN_SEC   = 1.0   # 같은 문장/고정 문구 연타 방지
PAIR_MAX_BACK  = 6     # 동사 앞에서 최대 몇 개 안에서 명사를 찾을지(거리 기반)

# ====== CORS ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== 정규화 유틸 ======
def to_text(x: Any) -> str:
    if x is None:
        s = ""
    elif isinstance(x, bytes):
        try:
            s = x.decode("utf-8", "ignore")
        except Exception:
            s = str(x)
    else:
        try:
            s = x.item()  # numpy scalar -> python
        except Exception:
            s = x
        s = str(s)
    # 모델이 NFD로 내면 NFC로 맞춘다(비교/조사/endswith 모두 정상화)
    return unicodedata.normalize("NFC", s.strip())

# ====== 라벨/품사 (NFC로 미리 정규화) ======
_raw_FIXED = {"안녕하세요", "감사합니다"}
_raw_NOUNS = {"열", "콧물", "코", "기침"}            # 필요시 확장
_raw_VERBS = {"있다", "없다", "막히다", "아프다"}    # 필요시 확장

FIXED_UTTERANCES = {to_text(s) for s in _raw_FIXED}
NOUN_OVERRIDES   = {to_text(s) for s in _raw_NOUNS}
VERB_OVERRIDES   = {to_text(s) for s in _raw_VERBS}

def is_verb(w: str) -> bool:
    if not isinstance(w, str):
        return False
    if w in VERB_OVERRIDES: return True
    if w in NOUN_OVERRIDES: return False
    # '다' 비교도 NFC에서만 정확히 작동
    return w.endswith("다")

def has_jongseong(word: str) -> bool:
    # 마지막 글자가 완성형 한글일 때만 정확
    if not word:
        return False
    ch = word[-1]
    base = ord('가')
    code = ord(ch) - base
    return 0 <= code <= 11171 and (code % 28) != 0

def subject_particle(noun: str) -> str:
    return '이' if has_jongseong(noun) else '가'

def format_noun_verb(noun: str, verb: str) -> str:
    if not noun or not verb: return ""
    return f"{noun}{subject_particle(noun)} {verb}"

# ====== 모델 ======
model, classes = load_model()
mp_holistic = mp.solutions.holistic

@app.get("/")
def root():
    return {"message": "Hello FastAPI"}

# ====== 버퍼에서 문장 조립 (거리 기반) ======
def try_make_sentence_from_buffer_by_distance(
    buf: List[str],
) -> Tuple[str, int]:
    """
    buf: ['코','있다','기침', ...] 처럼 확정 단어만 순서대로 쌓인 리스트
    규칙:
      1) 뒤에서부터 '가장 최근 동사'의 인덱스를 v_idx로 잡음
      2) v_idx 바로 앞쪽 범위에서(최대 PAIR_MAX_BACK 개) 가장 가까운 명사 n_idx를 찾음
      3) 찾으면 문장 만들고 buf[:v_idx+1] 소비
    반환: (sentence, consume_upto) / 실패 시 ("", 0)
    """
    if not buf:
        return "", 0

    # 1) 최근 동사
    v_idx = -1
    for i in range(len(buf) - 1, -1, -1):
        if is_verb(buf[i]):
            v_idx = i
            break
    if v_idx == -1:
        return "", 0

    # 2) v_idx 앞에서 가까운 명사
    start = max(0, v_idx - PAIR_MAX_BACK)
    n_idx = -1
    for j in range(v_idx - 1, start - 1, -1):
        if not is_verb(buf[j]):
            n_idx = j
            break
    if n_idx == -1:
        return "", 0

    noun = buf[n_idx]
    verb = buf[v_idx]
    sentence = format_noun_verb(noun, verb)
    if not sentence:
        return "", 0

    return sentence, (v_idx + 1)  # 동사까지 소비

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 연결됨")

    sequence = []
    recent_preds = deque(maxlen=WIN)  # 예측 창
    word_buffer: List[str] = []       # 확정 단어만 저장 (정규화된 str)

    last_confirm_time = 0.0
    last_sentence_time = 0.0
    last_sentence_text = ""
    boot_time = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        try:
            while True:
                try:
                    base64_data = await websocket.receive_text()
                    frame = decode_base64_image(base64_data)

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    keypoints = extract_keypoints(results)

                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    resp = {"coordinates": keypoints.tolist()}

                    if len(sequence) == 30:
                        raw = predict_from_keypoints(np.array(sequence), model, classes)
                        pred = to_text(raw)          # ★ NFC 정규화
                        resp["live"] = pred

                        now = time.time()

                        # 1) 고정 문구는 즉시 문장 출력(쿨다운 적용)
                        if pred in FIXED_UTTERANCES:
                            if (now - last_sentence_time) >= COOLDOWN_SEC or last_sentence_text != pred:
                                resp["sentence"] = pred
                                last_sentence_time = now
                                last_sentence_text = pred
                            # 버퍼/창 초기화
                            recent_preds.clear()
                            word_buffer.clear()
                            last_confirm_time = 0.0
                            await websocket.send_text(json.dumps(resp))
                            continue

                        # 2) 일반 단어는 다수결로 확정 → 버퍼 push
                        recent_preds.append(pred)
                        if len(recent_preds) >= 3:  # 너무 초기 튐 방지
                            cnt = Counter(recent_preds)
                            top_word, top_count = cnt.most_common(1)[0]
                            if top_count >= MAJ:
                                # 연속 동일 push 방지
                                if not word_buffer or word_buffer[-1] != top_word:
                                    word_buffer.append(top_word)
                                    last_confirm_time = now
                                    # 확정 직후 즉시 조립 시도
                                    sentence, consume = try_make_sentence_from_buffer_by_distance(word_buffer)
                                    if sentence:
                                        if (now - last_sentence_time) >= COOLDOWN_SEC or last_sentence_text != sentence:
                                            resp["sentence"] = sentence
                                            last_sentence_time = now
                                            last_sentence_text = sentence
                                        if consume > 0:
                                            del word_buffer[:consume]
                                        recent_preds.clear()  # 잔상 제거

                        # 3) 타임아웃 시 강제 조립 한 번 더 시도 → 안 되면 초기화
                        if last_confirm_time:
                            inactive = (now - last_confirm_time) > INACTIVITY_SEC
                            hard = (now - boot_time) > HARD_RESET_SEC and inactive
                            if inactive or hard:
                                sentence, consume = try_make_sentence_from_buffer_by_distance(word_buffer)
                                if sentence:
                                    if (now - last_sentence_time) >= COOLDOWN_SEC or last_sentence_text != sentence:
                                        resp["sentence"] = sentence
                                        last_sentence_time = now
                                        last_sentence_text = sentence
                                    if consume > 0:
                                        del word_buffer[:consume]
                                # 소비 후 남은 게 없으면 완전 초기화
                                if not word_buffer:
                                    recent_preds.clear()
                                    last_confirm_time = 0.0

                    await websocket.send_text(json.dumps(resp))

                except Exception as e:
                    print("loop error:", e)
                    continue

        except WebSocketDisconnect:
            print("WebSocket 연결 종료")
