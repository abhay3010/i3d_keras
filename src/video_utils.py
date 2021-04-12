import cv2
import pathlib
import numpy as np
"""
Crops the centre of the image square. 
Taken from https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
"""
def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]
"""
returns the video frames for a given video file path
"""
def get_video_frames(video_filepath, resize=True,  resize_shape=(112, 112)):
    p = pathlib.Path(video_filepath)
    if not p.exists():
        raise ValueError("Filepath not found {0}".format(video_filepath))
    frames = []
    cap = cv2.VideoCapture(str(video_filepath))
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resize and frame.shape != resize_shape:
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_CUBIC)
            frames.append(frame)
    finally:
        cap.release()
    return np.array(frames)
