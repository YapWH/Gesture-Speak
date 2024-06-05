import torch
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from torchvision.datasets import ImageFolder
from torch import nn, optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder("./data/train", transform=transform)
    validation_dataset = ImageFolder("./data/validation", transform=transform)
    test_dataset = ImageFolder("./data/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EfficientNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    train(model, criterion, optimizer, train_loader, validation_loader, num_epochs)
    test(model, criterion, test_loader)
