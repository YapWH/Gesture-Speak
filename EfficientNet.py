import cv2
import torch
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import os

# Load the model
model = efficientnet_v2_s(pretrained=False, num_classes=1000)
model.eval()

# Define transformation for the frames
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Function to preprocess each frame
def preprocess_frame(frame):
    # Convert frame to PIL Image
    frame_pil = Image.fromarray(frame)

    # Apply the transform to the frame
    frame_t = transform(frame_pil)
    frame_t = frame_t.unsqueeze(0)  # Add a batch dimension
    return frame_t

# Preprocess frames from processed_frames directory
processed_frames_folder = "./data"
frames = []
for filename in os.listdir(processed_frames_folder):
    if filename.endswith(".jpg"):
        frame_path = os.path.join(processed_frames_folder, filename)
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)  # Read frame RGB
        frame_tensor = preprocess_frame(frame)
        frames.append(frame_tensor)

# Process frames through the model
with torch.no_grad():
    for frame_tensor in frames:
        output = model(frame_tensor)

        print(output)
