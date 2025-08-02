from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np

# 모델 로드 
from model.predict import load_model, predict_from_keypoints

app = FastAPI()

# CORS 설정 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, classes = load_model()
last_prediction = None

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 연결됨")

    last_prediction=None 
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            coordinates = message.get("coordinates", [])
            print(f"받은 데이터 프레임 수: {len(coordinates)}")

            if isinstance(coordinates, list) and len(coordinates) == 30:
                valid_frames = [frame for frame in coordinates if isinstance(frame, list) and len(frame) == 154]
                if len(valid_frames) == 30:
                    try:
                        keypoints = np.array(valid_frames)  # (30, 154)
                        print("예측 시작 - shape:", keypoints.shape)
                        predicted_text = predict_from_keypoints(keypoints, model)
                        await websocket.send_text(json.dumps({"text": predicted_text}))
                        print("예측 단어:", predicted_text)

                        # 중복 예측 억제
                        if predicted_text != last_prediction:
                            await websocket.send_text(json.dumps({"text": predicted_text}))
                            last_prediction = predicted_text
                        else:
                            print("중복 단어 - 전송 생략")

                    except Exception as e:
                        print("예측 중 오류:", e)
                        await websocket.send_text(json.dumps({"text": "예측 실패"}))
                else:
                    print(f"좌표 수 오류: {len(coordinates)}개")
                    await websocket.send_text(json.dumps({"text": "좌표 개수 오류: 30 x 154 형식 아님"}))

    except WebSocketDisconnect:
        print("WebSocket 연결 종료")