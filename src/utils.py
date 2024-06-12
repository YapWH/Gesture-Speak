from moviepy.editor import VideoFileClip
import numpy as np

def trim_videos(video:str, start_time:float=0, end_time:float=2):
    """
    Cuts videos in a folder to a specified clip duration.

    Args:
        video: Path to the video file.
        clip_duration: Duration of the clip in seconds. Defaults to 2.
    """
    clip = VideoFileClip(video)

    if clip.duration < (end_time - start_time):
        clip_duration = clip.duration
        frames = clip.subclip(0, clip_duration)
    else:
        frames = clip.subclip(start_time, end_time)

    return frames
