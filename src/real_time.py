import cv2
import torch
import pickle
from time import time
import torchvision.transforms as transforms

from Sign_Language import EfficientNetSmall, NGramModel

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") # For testing purposes

################################################################################

@torch.no_grad()
def predict(model, ngram, frame, dataset_classes, nn_weights:float=0.5, ngram_weights:float=0.5):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    frame = transform(frame).unsqueeze(0).to(device)
    
    model.eval()
    predicted_sequence = ['<s>']
    
    outputs = model(frame)
    nn_probs = torch.softmax(outputs, dim=1)
    _, nn_predicted = torch.max(outputs, 1)
    predicted_labels = [dataset_classes[label] for label in nn_predicted]

    for i, label in enumerate(predicted_labels):
        ngram_predictions = ngram.predict(label)
        ngram_index = dataset_classes.index(ngram_predictions)

        combined_probs = nn_weights * nn_probs[i] + ngram_weights * (torch.eye(len(dataset_classes))[ngram_index].to(device))
        corrected_label_index = torch.argmax(combined_probs).item()
        corrected_label = dataset_classes[corrected_label_index]
        predicted_sequence[-1] = corrected_label
    
    return ''.join(predicted_sequence)

################################################################################

def real_time():
    model = EfficientNetSmall().to(device)
    model.load_state_dict(torch.load("../src/model/student_model.pth"))
    n_gram = NGramModel().load("../src/model/ngram_model.pkl")
    dataset_classes = pickle.load(open("../src/model/dataset_classes.pkl", "rb"))

    cv2.namedWindow("Sign-Language Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sign-Language Recognition", 400, 400)
    cap  = cv2.VideoCapture(0)
    if not cap:
        raise Exception("Error opening video capture")
    
    start = time()
    
    while True:
        ret, frame = cap.read()

        if not ret:
            raise Exception("Error reading video frame")
        
        cv2.imshow("Frame", frame)
    
        if (time() - start) >= 1:
            print(predict(model, n_gram, frame, dataset_classes, nn_weights=0.7, ngram_weights=0.3))
            start = time()

        # Exit 
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

################################################################################

if __name__ == "__main__":
    real_time()