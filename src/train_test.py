import os
import shutil
import random
import pickle
import torch
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from torchvision.datasets import ImageFolder
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import defaultdict, Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################

def check_and_delete_empty_directories(root_dir):
    for cls in os.listdir(root_dir):
        cls_dir = os.path.join(root_dir, cls)
        if os.path.isdir(cls_dir):
            contains_jpg = any(file.endswith('.jpg') for file in os.listdir(cls_dir))
            if not contains_jpg:
                print(f"Deleting directory: {cls_dir}")
                shutil.rmtree(cls_dir)

class SequenceImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        instances = []
        directory = os.path.expanduser(directory)
        for target_class in sorted(class_to_idx.keys()):
            class_idx = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            class_file = os.path.join(target_dir, f"{target_class}.jpg")
            if os.path.isfile(class_file):
                item = (class_file, class_idx)
                instances.append(item)
        return instances

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

################################################################################

class EfficientNet(nn.Module):
    def __init__(self):
        """
        Initializes the EfficientNet model.

        Args:
            None

        Returns:
            None
        """
        super(EfficientNet, self).__init__()
        self.base_model = efficientnet_v2_s(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(num_features, len(train_dataset.classes))

    def forward(self, x):
        """
        The forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.base_model(x)

def train(model, criterion, optimizer, train_loader, validation_loader, num_epochs):
    """
    Train the model on the training set.

    Args:
        model: PyTorch model.
        criterion: Loss function.
        optimizer: Optimizer.
        train_loader: Dataloader containing the training set.
        validation_loader: Dataloader containing the validation set.
        num_epochs: Number of epochs to train the model.

    Returns:
        List of training losses.
    """
    train_loss = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        cur_loss = running_loss/len(train_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {cur_loss:.3f} ")

        train_loss.append(cur_loss)
    
    return train_loss

@torch.no_grad()
def test(model, criterion, test_loader):
    """
    Tests the model on the test set.

    Args:
        model: PyTorch model.
        criterion: Loss function.
        test_loader: Dataloader containing the test set.

    Returns:
        Test accuracy and loss.
    """
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    
    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}, Loss: {loss:.4f}")

    return test_acc, loss

################################################################################

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

################################################################################

if __name__ == "__main__":

    # Neural Network
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_root = "./autodl-tmp/SL/ASL_Citizen/train"
    validation_root = "./autodl-tmp/SL/ASL_Citizen/validation"
    test_root = "./autodl-tmp/SL/ASL_Citizen/test"
    
    check_and_delete_empty_directories(train_root)
    check_and_delete_empty_directories(validation_root)
    check_and_delete_empty_directories(test_root)

    train_dataset = SequenceImageFolder(train_root, transform=transform)
    validation_dataset = SequenceImageFolder(validation_root, transform=transform)
    test_dataset = SequenceImageFolder(test_root, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EfficientNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    train(model, criterion, optimizer, train_loader, validation_loader, num_epochs)
    test(model, criterion, test_loader)

    torch.save(model.state_dict(), 'model.pth')

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