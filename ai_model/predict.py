import os
import numpy as np
import torch
from .model import SignLanguageBiLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_bilstm_val_100_20250728.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "classes.npy")

classes = np.load(CLASSES_PATH, allow_pickle=True)

def load_model():
    model = SignLanguageBiLSTM(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, classes

def predict_from_keypoints(keypoints_30x258, model, classes):
    if isinstance(keypoints_30x258, list):
        keypoints_30x258 = np.array(keypoints_30x258)
    if keypoints_30x258.shape != (30, 258):
        raise ValueError(f"❗ 입력 shape 오류: 기대값은 (30, 258), 현재는 {keypoints_30x258.shape}")
    input_tensor = torch.tensor(keypoints_30x258, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        return classes[pred_idx]
