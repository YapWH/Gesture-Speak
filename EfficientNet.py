import torch
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from torchvision.datasets import ImageFolder
from torch import nn, optim
from torch.utils.data import DataLoader

# Define hyperparameters
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Define transformation for the frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Load your custom dataset using ImageFolder
train_dataset = ImageFolder("./data/train", transform=transform)  # Load train dataset
validation_dataset = ImageFolder("./data/validation", transform=transform)  # Load validation dataset
test_dataset = ImageFolder("./data/test", transform=transform)  # Load test dataset

# Create DataLoader instances for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the pre-trained model
model = efficientnet_v2_s(pretrained=True)

# Freeze pre-trained layers (optional)
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers

# Modify the final layer for your number of classes
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(train_dataset.classes))  

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(model, criterion, optimizer, train_loader, validation_loader, num_epochs):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Train phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
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
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {running_loss/len(train_loader):.3f} - Val Acc: {val_acc:.4f}")

        # Save model with best validation accuracy (optional)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")


# Define the test function
def test(model, criterion, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    
# Train the model
train(model, criterion, optimizer, train_loader, validation_loader, num_epochs)
test(model, criterion, test_loader)
