import os
import shutil
import torch
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from torchvision.datasets import ImageFolder
from torch import nn, optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
def check_and_delete_empty_directories(root_dir):
    for cls in os.listdir(root_dir):
        cls_dir = os.path.join(root_dir, cls)
        if os.path.isdir(cls_dir):
            contains_jpg = any(file.endswith('.jpg') for file in os.listdir(cls_dir))
            if not contains_jpg:
                print(f"Deleting directory: {cls}")
                os.rmdir(cls_dir)
                
class SequenceImageFolder(ImageFolder):
    def __init__(self, root, transform=None, num_frames_per_sequence=4):
        super().__init__(root, transform=transform)
        self.num_frames_per_sequence = num_frames_per_sequence

    def __getitem__(self, index):
        path, _ = self.samples[index]
        target = self.targets[index]
        sample = self.loader(path)

        sequence = []
        for i in range(self.num_frames_per_sequence):
            frame = self.transform(sample)
            sequence.append(frame)

        # Combine frames into a sequence
        sequence = torch.stack(sequence, dim=0)

        return sequence, target

################################################################################

class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.base_model = efficientnet_v2_s(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(num_features, len(train_dataset.classes))

    def forward(self, x):
        return self.base_model(x)

################################################################################

# Training loop
def train(model, criterion, optimizer, train_loader, validation_loader, num_epochs):
    best_val_acc = 0.0
    train_loss = []
    train_acc = []
    early_stopping = 5
    for epoch in range(num_epochs):
        # Train phase
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
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        cur_loss = running_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {cur_loss:.3f} - Val Acc: {val_acc:.4f}")

        train_loss.append(cur_loss)
        train_acc.append(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_{epoch}.pth")
        else:
            if epoch - early_stopping > 0:
                print("Early stopping")
                break
            print(f"Model not improving for {epoch - early_stopping} epochs")

################################################################################

def test(model, criterion, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
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

if __name__ == "__main__":

    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10
    num_frames_per_sequence = 4
    
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_root = "./autodl-tmp/SL/ASL_Citizen/train"
    validation_root = "./autodl-tmp/SL/ASL_Citizen/validation"
    test_root = "./autodl-tmp/SL/ASL_Citizen/test"
    
    check_and_delete_empty_directories(train_root)
    check_and_delete_empty_directories(validation_root)
    check_and_delete_empty_directories(test_root)

    train_dataset = SequenceImageFolder(train_root, transform=transform, num_frames_per_sequence=num_frames_per_sequence)
    validation_dataset = SequenceImageFolder(validation_root, transform=transform, num_frames_per_sequence=num_frames_per_sequence)
    test_dataset = SequenceImageFolder(test_root, transform=transform, num_frames_per_sequence=num_frames_per_sequence)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EfficientNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    train(model, criterion, optimizer, train_loader, validation_loader, num_epochs)
    test(model, criterion, test_loader)
