from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import mediapipe as mp

# 모델 로드 
from ai_model.predict import load_model, predict_from_keypoints
from ai_model.preprocessing import decode_base64_image, extract_keypoints

app = FastAPI()

# CORS 설정 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 초기화 
model, classes = load_model()
mp_holistic = mp.solutions.holistic

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 연결됨")

    sequence = []
    last_prediction = None
    recent_predictions = []
    MAX_QUEUE = 5
    THRESHOLD_COUNT = 3

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        try:
            while True:
                # 프론트에서 base64 프레임 수신
                base64_data = await websocket.receive_text()
                frame = decode_base64_image(base64_data)

                # Mediapipe 처리
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                keypoints = extract_keypoints(results)  # shape = (258,)

                sequence.append(keypoints)
                sequence = sequence[-30:]

                # 시퀀스가 30프레임 쌓이면 예측
                if len(sequence) == 30:
                    prediction = predict_from_keypoints(np.array(sequence), model, classes)

                    recent_predictions.append(prediction)
                    recent_predictions = recent_predictions[-MAX_QUEUE:]
                    most_common = max(set(recent_predictions), key=recent_predictions.count)

                    if recent_predictions.count(most_common) >= THRESHOLD_COUNT and most_common != last_prediction:
                        await websocket.send_text(json.dumps({"result": most_common}))
                        last_prediction = most_common

        except WebSocketDisconnect:
            print("🔌 WebSocket 연결 종료")
        except Exception as e:
            print("예측 중 오류:", e)
            await websocket.send_text(json.dumps({"error": "예측 실패"}))