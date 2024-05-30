import cv2
from utils import cut_videos
import os

def preprocess_video(video_path:str, num_frames:int=4):
    """
    Preprocesses a video by selecting every nth frame and converting it to grayscale.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to skip. Defaults to 4.
    """
    count = 0
    selected_frames = []
    trimmed_video = cut_videos(video_path)
    video = cv2.VideoCapture(video_path)

    if not video.isOpened(): raise Exception("Error opening video file")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = video.read()
        if not ret: break
        count += 1

        if count % (total_frames//num_frames) == 0:
            height, width = frame.shape[:2]
            min_dim = min(height, width)
            center_x = (width - min_dim) // 2
            center_y = (height - min_dim) // 2
            square_frame = frame[center_y:center_y+min_dim, center_x:center_x+min_dim]

            resized_frame = cv2.resize(square_frame, (224, 224))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            selected_frames.append(gray_frame)

        if len(selected_frames) == num_frames:
            return selected_frames


if __name__ == "__main__":
    video_folder = "./data"
    for filename in os.listdir(video_folder):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_folder, filename)
            frames= preprocess_video(video_path)

    for i in range(len(frames)):
        cv2.imwrite(f"./data/frame_{i}.jpg", frames[i])
