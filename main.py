from fastapi import FastAPI, WebSocket, WebSocketDisconnect 
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import mediapipe as mp
import cv2
import time
from collections import deque, Counter

from ai_model.predict import load_model, predict_from_keypoints
from ai_model.preprocessing import decode_base64_image, extract_keypoints
from ai_model.gpt_utils import complete_sentence  # GPT 문장 생성 함수 추가

app = FastAPI()

# CORS 설정: 프론트엔드와의 통신을 위해 모든 origin 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_verb(w: str) -> bool:
    return isinstance(w, str) and w.endswith("다")

# 모델 및 클래스 리스트 로드
model, classes = load_model()
mp_holistic = mp.solutions.holistic


@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 연결됨")

    # 30프레임짜리 시퀀스를 구성하기 위한 리스트
    sequence = []

    # 최근 예측값 저장용 큐 (5개)
    recent_predictions = []
    MAX_QUEUE = 5
    THRESHOLD_COUNT = 3  # 동일 예측이 3번 이상일 때 확정 단어로 간주

    # 문장 인식 관련 상태 변수들
    word_buffer = []  # 문장에 포함될 단어들을 순서대로 저장
    confirmed_words = deque(maxlen=15)  # 최근 예측된 단어 저장 (문장 종료 판단용)
    # 조건: 단어가 많으면 적게 반복되어도 문장 종료 가능
    min_repeat_dynamic = max(5, 12 - len(set(word_buffer)))    
    last_confirmed_word = None  # 마지막으로 확정된 단어
    last_sentence_time = 0  # 가장 최근 문장 종료 시각

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        try:
            while True:
                try:
                    # 프론트에서 base64 인코딩된 이미지 수신
                    base64_data = await websocket.receive_text()
                    frame = decode_base64_image(base64_data)

                    # 프레임을 RGB로 변환 후 Mediapipe로 키포인트 추출
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image_rgb)
                    keypoints = extract_keypoints(results)
                    # print("keypoints shape:", keypoints.shape)

                    # 추출된 keypoints를 시퀀스에 추가
                    sequence.append(keypoints)
                    sequence = sequence[-30:]  # 시퀀스 길이를 30으로 고정
                    # 기본 응답: 좌표는 항상 보냄
                    response = {"coordinates": keypoints.tolist()}

                    # 시퀀스가 30개 이상 쌓이면 예측 수행
                    if len(sequence) == 30:
                        prediction = predict_from_keypoints(np.array(sequence), model, classes)
                        recent_predictions.append(prediction)
                        recent_predictions = recent_predictions[-MAX_QUEUE:]

                        # 현재 큐에서 자주 등장한 단어 찾기
                        counter = Counter(recent_predictions)
                        common_words = counter.most_common()
                        most_common_word = prediction; sentence_to_send = None; count = 0

                        reliable_words = [w for w, c in common_words if c >= THRESHOLD_COUNT]
                        if reliable_words:
                                word = reliable_words[0]
                                if word != last_confirmed_word:
                                    word_buffer.append(word)
                                    last_confirmed_word = word
                                confirmed_words.append(word)
                                counter = Counter(confirmed_words)
                                most_common_word, count = counter.most_common(1)[0]
                        
                        # 단어 예측 결과가 있으면 포함
                        if most_common_word:
                            response["result"] = most_common_word
                        
                        # 버퍼 내 단어 수에 따라 반복 조건 완화
                        min_repeat_dynamic = max(5, 12 - len(set(word_buffer)))  # 예: 단어 2개면 min_repeat=10, 3개면 9 ...


                        # 문장 종료 조건 판단
                        sentence_to_send = None
                        if count >= min_repeat_dynamic and time.time() - last_sentence_time > 3:
                            # 최근 확정 단어에서 역순으로 서로 다른 2개만 취득
                            uniq = []
                            for w in reversed(word_buffer):
                                if w not in uniq:
                                    uniq.append(w)
                                if len(uniq) == 2:
                                    break
                            pair = list(reversed(uniq))  # 시간 순서 유지
                                                        
                            # 동사 2개 금지: 한 번만 허용
                            filtered = []
                            verb_seen = False
                            for t in pair:
                                if is_verb(t):
                                    if verb_seen: continue
                                    verb_seen = True
                                filtered.append(t)
                            pair = filtered[:2]

                            # 명사→동사 정렬: 앞이 동사이고 뒤가 동사가 아니면 스왑
                            if len(pair) == 2 and is_verb(pair[0]) and not is_verb(pair[1]):
                                pair = [pair[1], pair[0]]
                            merged = " ".join(pair[:2])  # 최대 2단어
                            # GPT 프롬프트: 1~2어절, 명사 우선, 동사 끝, 새 단어 금지
                            prompt = (
                                "다음 단어들을 1~2개 어절로 정렬해 출력해줘. "
                                "명사는 먼저, 동사는 마지막. 조사는 최소화. 새로운 단어 추가 금지. 따옴표 없이:\n"
                                f"{merged}"
                            )
                             
                            used_gpt = False
                            try:
                                sentence_to_send = complete_sentence(prompt)
                                used_gpt = True          

                            except Exception as e:
                                print("GPT error:", e)
                                sentence_to_send = merged  # 실패 시 단어 합친 문장으로 대체

                            print("문장 완성:", sentence_to_send)

                            # 상태 초기화
                            word_buffer.clear()
                            confirmed_words.clear()
                            last_confirmed_word = None
                            last_sentence_time = time.time()

                        # 문장 있으면 필드 추가
                        if sentence_to_send:
                            response["sentence"] = sentence_to_send
                            response["from_gpt"] = used_gpt
                            response["sent_to_gpt"] = merged

                    # 좌표는 매 프레임 전송
                    await websocket.send_text(json.dumps(response))
                except Exception as e:
                    print("loop error:", e)
                    continue

        except WebSocketDisconnect:
            print("WebSocket 연결 종료")
