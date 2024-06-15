import random
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
from collections import defaultdict, Counter
from tqdm import tqdm
from utils import set_logger
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################################################

class SignLanguageDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels.astype(np.int64) 
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.features[idx].astype(np.uint8)
        image = np.stack([image] * 3, axis=-1)  # Convert to 3 channels
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
    
#########################################################################
class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
        self.base_model = efficientnet_v2_l(weights=weights)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

def train(model, criterion, optimizer, train_loader, val_loader, num_epochs, patience):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    with tqdm(range(num_epochs), desc="Training") as pbar:
        for epoch in pbar:
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            val_loss, val_accuracy = test(model, criterion, val_loader)
            val_losses.append(val_loss)

            pbar.set_postfix({'Train Loss': train_loss, 'Val Loss': val_loss, 'Val Accuracy': val_accuracy})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), './model/best_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logging.info(f'Early stopping at epoch {epoch}')
                break

    return train_losses, val_losses


@torch.no_grad()
def test(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    accuracy = correct / total
    
    return test_loss, accuracy

#########################################################################

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(Counter)
    
    def train(self, sequences):
        for seq in sequences:
            padded_seq = ['<s>'] * (self.n - 1) + list(seq) + ['</s>']
            for i in range(len(padded_seq) - self.n + 1):
                context = tuple(padded_seq[i:i+self.n-1])
                target = padded_seq[i+self.n-1]
                self.ngrams[context][target] += 1
    
    def predict(self, context):
        context = tuple(context[-(self.n-1):])
        if context in self.ngrams:
            return self.ngrams[context].most_common(1)[0][0]
        else:
            return random.choice(list(self.ngrams.keys()))[-1]
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

def load_external_sequences(external_dataset_path):
    sequences = []
    with open(external_dataset_path, 'r') as f:
        for line in f:
            # Tokenize the line into words (or signs)
            sequences.append(line.strip())
    return sequences

def predict_sequence(model, ngram_model, dataloader, dataset_classes, nn_weight=0.5, ngram_weight=0.5):
    model.eval()
    predicted_sequence = []
    for inputs, _ in dataloader:
        outputs = model(inputs.to(device))
        nn_probs = torch.softmax(outputs, dim=1)
        _, nn_predicted = torch.max(outputs, 1)
        predicted_labels = [dataset_classes[label] for label in nn_predicted.cpu().numpy()]
        
        for i, label in enumerate(predicted_labels):
            predicted_sequence.append(label)
            if len(predicted_sequence) >= ngram_model.n - 1:
                ngram_prediction = ngram_model.predict(predicted_sequence)
                ngram_index = dataset_classes.index(ngram_prediction)

                # Combine NN and n-gram predictions using the weights
                combined_probs = nn_weight * nn_probs[i] + ngram_weight * (torch.eye(num_classes)[ngram_index].to(nn_probs.device))
                corrected_label_index = torch.argmax(combined_probs).item()
                corrected_label = dataset_classes[corrected_label_index]
                predicted_sequence[-1] = corrected_label
                
    return ''.join(predicted_sequence)

#########################################################################

def knowledge_distillation_loss(teacher_logits, student_logits, true_labels, alpha=0.5, beta=0.5):
    soft_target_loss = F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1), reduction='batchmean')
    mse_loss = F.mse_loss(student_logits, F.one_hot(true_labels, num_classes=student_logits.size(1)).float(), reduction='mean')
    distillation_loss = alpha * soft_target_loss + beta * mse_loss
    
    return distillation_loss

def train_student(student_model, teacher_model, device, train_loader, optimizer, alpha, beta, epochs):
    
    student_model.train()
    teacher_model.eval()

    train_loss = []

    with tqdm(range(epochs), desc="Training") as pbar:
        for epoch in pbar:
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                teacher_outputs = teacher_model(inputs)
                student_outputs = student_model(inputs)
                loss = knowledge_distillation_loss(teacher_outputs, student_outputs, labels, alpha, beta)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loss.append(running_loss / len(train_loader))
            pbar.set_postfix({'Train Loss': train_loss[-1]})
    
    return train_loss

class EfficientNetSmall(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetSmall, self).__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.base_model = efficientnet_v2_s(weights=weights)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

#########################################################################

if __name__ == "__main__":
    set_logger(f"./log/{time}.log")

    # Neural Network
    learning_rate = 0.001
    batch_size = 16
    num_epochs = 50
    patience = 50  # Early stopping patience

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
     # Load dataset
    dataset = ImageFolder(root='../asl_dataset', transform=transform)
    
    # Split dataset into training, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(dataset.classes)
    
    model = EfficientNet(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    train_losses, val_losses = train(model, criterion, optimizer, train_loader, val_loader, num_epochs, patience)
    test_loss, test_accuracy = test(model, criterion, test_loader)
    logging.info(f'Test Accuracy: {test_accuracy:.4f} - Loss: {test_loss:.4f}')

    torch.save(model.state_dict(), './model/model.pth')

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.savefig('loss_curve.png')
    plt.show()  # Save the loss curves plot

    # N-gram
    teacher_model = EfficientNet(num_classes=num_classes).to(device)
    teacher_model.load_state_dict(torch.load('./model/model.pth'))
    
    num_classes = len(dataset.classes)
    external_sequences = load_external_sequences("./data/data.txt")
    ngram_model = NGramModel(n=2)
    ngram_model.train(external_sequences)

    sequences = []
    for inputs, labels in train_loader:
        _, predicted = torch.max(teacher_model(inputs.to(device)), 1)
        sequences.append([dataset.classes[label] for label in predicted])
    ngram_model.train(sequences)

    ngram_model.save("ngram_model.pkl")

    ngram_model = NGramModel.load("ngram_model.pkl")

    pickle.dump(dataset.classes, open("classes.pkl", "wb"))

    # predicted_sequences = predict_sequence(model, ngram_model, test_loader, dataset.classes, ngram_weight=0.3, nn_weight=0.7)
    # print(predicted_sequences)

    student_model = EfficientNetSmall(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(student_model.parameters(), lr=learning_rate)

    train_student(student_model, teacher_model, device, train_loader, optimizer, alpha=0.5, beta=0.5, epochs=50)
    torch.save(model.state_dict(), './model/student_model.pth')
    test_loss, test_accuracy = test(model, criterion, test_loader)
    logging.info(f'Test Accuracy: {test_accuracy:.4f} - Loss: {test_loss:.4f}')