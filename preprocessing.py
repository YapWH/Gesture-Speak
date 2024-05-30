import cv2
import os
from utils import trim_videos

def preprocess_video(video_path: str, num_frames: int = 4, start_time: float = 0, end_time: float = 2):
    """
    Preprocesses a video by selecting every nth frame and converting it to grayscale.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to select. Defaults to 4.
        start_time: Start time for trimming the video in seconds. Defaults to 0.
        end_time: End time for trimming the video in seconds. Defaults to 2.
    """
    # Trim Video to 2 sec
    trimmed_clip = trim_videos(video_path, start_time, end_time)
    fps = trimmed_clip.fps
    total_frames = int(trimmed_clip.duration * fps)

    # Read Video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened(): raise Exception("Error opening video file")
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time else total_frames
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_step = total_frames // num_frames if total_frames >= num_frames else 1
    count = 0
    selected_frames = []

    while True:
        ret, frame = video.read()
        if not ret or count >= end_frame - start_frame: break

        if count % frame_step == 0:
            height, width = frame.shape[:2]
            min_dim = min(height, width)
            center_x = (width - min_dim) // 2
            center_y = (height - min_dim) // 2
            square_frame = frame[center_y:center_y + min_dim, center_x:center_x + min_dim]

            resized_frame = cv2.resize(square_frame, (224, 224))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            selected_frames.append(gray_frame)

            if len(selected_frames) == num_frames:
                break

        count += 1

    video.release()
    return selected_frames


if __name__ == "__main__":
    video_folder = "./data"
    for filename in os.listdir(video_folder):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_folder, filename)
            frames= preprocess_video(video_path)

    for i in range(len(frames)):
        cv2.imwrite(f"./data/frame_{i}.jpg", frames[i])
