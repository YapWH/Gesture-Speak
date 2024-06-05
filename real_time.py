import torch
import torch.functional as F
import cv2
from time import time
from train_test import EfficientNet

from preprocessing import preprocess_video

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Real time
def predict(model, frame):
    frame = torch.tensor(frame).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(frame)
        largest_prob = F.softmax(outputs, dim=1).max().item()
        if largest_prob > 0.5:
            return outputs.argmax(dim=1).item()
        else:
            return ""

################################################################################

def real_time():
    model = EfficientNet().to(device)
    model.load_state_dict(torch.load("best_model.pth"))

    cap  = cv2.VideoCapture(0)
    if not cap:
        raise Exception("Error opening video capture")
    
    start = time()
    frames = []
    
    while True:
        ret, frame = cap.read()

        if not ret:
            raise Exception("Error reading video frame")
        
        frames.append(frame)
        cv2.imshow("Frame", frame)
    
        if (time() - start) >= 2:
            frame = preprocess_video(frame)
            predict(frame)
            start = time()

        # Exit 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()