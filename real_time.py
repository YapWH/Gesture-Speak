import cv2
import torch

from time import time

from train_test import EfficientNet, NGramModel
from preprocessing import preprocess_video

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################

@torch.no_grad()
def predict(model, ngram, frame, dataset_classes, nn_weights:float=0.5, ngram_weights:float=0.5):
    frame = torch.tensor(frame).unsqueeze(0).to(device)
    model.eval()
    
    outputs = model(frame)
    nn_probs = torch.softmax(outputs, dim=1)
    _, nn_predicted = torch.max(outputs, 1)
    predicted_labels = [dataset_classes[label] for label in nn_predicted]

    for i, label in enumerate(predicted_labels):
        ngram_predictions = ngram.predict(label)
        ngram_index = dataset_classes.index(ngram_predictions)

        combined_probs = nn_weights * nn_probs[i] + ngram_weights * (torch.eye(len(dataset_classes))[ngram_index].to(device))

    largest_prob, _ = torch.max(combined_probs)

    if largest_prob > 0.5:
        return torch.argmax(combined_probs).item()
    else:
        return ""

################################################################################

def real_time():
    model = EfficientNet().to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    n_gram = NGramModel()

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