import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

model = models.efficientnet_v2_s(weights=None)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

url = 'https://letsenhance.io/static/8f5e523ee6b2479e26ecc91b9c25261e/1015f/MainAfter.jpg'  
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert('RGB')

# Apply the transform to the image
img_t = transform(img)
img_t = img_t.unsqueeze(0)  # Add a batch dimension

with torch.no_grad():
    output = model(img_t)

print(output)
