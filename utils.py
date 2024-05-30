from moviepy.editor import VideoFileClip

def cut_videos(video, clip_duration=2):
    """
    Cuts videos in a folder to a specified clip duration.

    Args:
        video: Path to the video file.
        clip_duration: Duration of the clip in seconds. Defaults to 2.
    """
    clip = VideoFileClip(video)

    if clip.duration < clip_duration:
        clip_duration = clip.duration    

    out_clip = clip.subclip(0, clip_duration)


    return out_clip.to_videofile(video, codec="libx264")