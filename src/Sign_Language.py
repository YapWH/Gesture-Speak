import torch
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from torch import nn, optim
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

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
        """
        Initializes the EfficientNet model.

        Args:
            num_classes (int): Number of output classes.

        Returns:
            None
        """
        super(EfficientNet, self).__init__()
        self.base_model = efficientnet_v2_s(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        The forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.base_model(x)

def train(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    """
    Train the model on the training set.

    Args:
        model: PyTorch model.
        criterion: Loss function.
        optimizer: Optimizer.
        train_loader: Dataloader containing the training set.
        val_loader: Dataloader containing the validation set.
        num_epochs: Number of epochs to train the model.

    Returns:
        train_losses: List of training losses.
        val_losses: List of validation losses.
        val_accuracies: List of validation accuracies.
    """
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
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
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.3f}')
        
        # Validation loop
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        print(f'Validation Loss: {val_loss:.3f} - Accuracy: {val_accuracy:.2f}%')
    
    return train_losses, val_losses, val_accuracies

@torch.no_grad()
def test(model, criterion, test_loader):
    """
    Tests the model on the test set.

    Args:
        model: PyTorch model.
        criterion: Loss function.
        test_loader: Dataloader containing the test set.

    Returns:
        test_acc: Test accuracy
        loss: loss
    """
    model.eval()
    correct = 0
    total = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    test_acc = correct / total
    print(f'Test Accuracy: {test_acc:.4f} - Loss: {loss:.4f}')
    
    return test_acc, loss

class NGramModel:
    def __init__(self, n):
        """
        Initializes an n-gram model.

        Args:
            n: Order of the n-gram model.

        Returns:
            None
        """
        self.n = n
        self.ngrams = defaultdict(Counter)
    
    def train(self, sequences):
        """
        Trains the n-gram model on a list of sequences.

        Args:
            sequences: List of sequences.
        
        Returns:
            None
        """
        for seq in sequences:
            padded_seq = ['<s>'] * (self.n - 1) + list(seq) + ['</s>']
            for i in range(len(padded_seq) - self.n + 1):
                context = tuple(padded_seq[i:i+self.n-1])
                target = padded_seq[i+self.n-1]
                self.ngrams[context][target] += 1
    
    def predict(self, context):
        """
        Predicts the next word given a context.

        Args:
            context: List of words in the context.

        Returns:
            Predicted word.
        """
        context = tuple(context[-(self.n-1):])
        if context in self.ngrams:
            return self.ngrams[context].most_common(1)[0][0]
        else:
            return random.choice(list(self.ngrams.keys()))[-1]
    
    def save(self, file_path):
        """
        Saves the n-gram model to a file.

        Args:
            file_path: Path to save the n-gram model file.

        Returns:
            None
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(file_path):
        """
        Loads an n-gram model from a file.

        Args:
            file_path: Path to the n-gram model file.
        
        Returns:
            NGramModel object.
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)

def load_external_sequences(external_dataset_path):
    """
    Loads sequences from an external dataset.

    Args:
        external_dataset_path: Path to the external dataset file.
    
    Returns:
        List of sequences.
    """
    sequences = []
    with open(external_dataset_path, 'r') as f:
        for line in f:
            # Tokenize the line into words (or signs)
            sequences.append(line.strip())
    return sequences

def predict_sequence(model, ngram_model, dataloader, dataset_classes, nn_weight=0.5, ngram_weight=0.5):
    """
    Predicts the sign language sequence using a combination of a neural network model and an n-gram model.

    Args:
        model: Neural network model.
        ngram_model: n-gram model.
        dataloader: PyTorch DataLoader object.
        dataset_classes: List of classes in the dataset.
        nn_weight: Weight for the neural network model.
        ngram_weight: Weight for the n-gram model.
    
    Returns:
        List of predicted sign language sequence.
    """
    model.eval()
    predicted_sequence = []
    for inputs, _ in dataloader:
        outputs = model(inputs)
        nn_probs = torch.softmax(outputs, dim=1)
        _, nn_predicted = torch.max(outputs, 1)
        predicted_labels = [dataset_classes[label] for label in nn_predicted]
        
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
if __name__ == "__main__":
    # Neural Network
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10
    
    # Load the CSV files
    train_csv_path = './Data/sign/sign_mnist_train.csv' 
    test_csv_path = './Data/sign/sign_mnist_train.csv' 

    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)

    # Separate features and labels
    train_labels = train_data['label'].values
    train_features = train_data.drop(columns=['label']).values / 255.0  # Normalize pixel values
    train_features = train_features.reshape(-1, 28, 28) # Reshape features to 28x28 images

    test_labels = test_data['label'].values
    test_features = test_data.drop(columns=['label']).values / 255.0  # Normalize pixel values
    test_features = test_features.reshape(-1, 28, 28) # Reshape features to 28x28 images
    
    # Split the training data into training and validation sets
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42)

    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Create datasets
    train_dataset = SignLanguageDataset(train_features, train_labels, transform=transform)
    val_dataset = SignLanguageDataset(val_features, val_labels, transform=transform)
    test_dataset = SignLanguageDataset(test_features, test_labels, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Number of classes
    num_classes = 25

    model = EfficientNet(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    train_losses, val_losses, val_accuracies = train(model, criterion, optimizer, train_loader, val_loader, num_epochs)
    test(model, criterion, test_loader)

    torch.save(model.state_dict(), 'model_26.pth')

    # Plot the loss curves and save the plots
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.savefig('loss_curves.png')  # Save the loss curves plot

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.savefig('accuracy_curve.png')  # Save the accuracy curve plot

    plt.show()

    # N-gram
    num_classes = len(train_dataset.classes)
    external_sequences = load_external_sequences("external_dataset.txt")
    ngram_model = NGramModel(n=2)
    ngram_model.train(external_sequences)

    sequences = []
    for inputs, labels in train_loader:
        _, predicted = torch.max(model(inputs), 1)
        sequences.append([train_dataset.classes[label] for label in predicted])
    ngram_model.train(sequences)

    ngram_model.save("ngram_model.pkl")

    ngram_model = NGramModel.load("ngram_model.pkl")

    pickle.dump(train_dataset.classes, open("classes.pkl", "wb"))

    predicted_sequences = predict_sequence(model, ngram_model, test_loader, train_dataset.classes, ngram_weight=0.3, nn_weight=0.7)
    print(predicted_sequences)