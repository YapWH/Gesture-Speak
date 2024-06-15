def trim_videos(video:str, start_time:float=0, end_time:float=2):
    from moviepy.editor import VideoFileClip

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



def set_logger(log_path):
    import os
    import logging

	# remove the log with same name
    if os.path.exists(log_path) is True:
        os.remove(log_path)
    if os.path.exists(log_path) is False:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Initialize log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)