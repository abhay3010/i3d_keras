import cv2
import pathlib
import numpy as np
import json
from numpy.lib.npyio import save
import tensorflow as tf
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
    parent_path = p.parents[0]
    saved_frames = []
    print(p)
    fname = p.stem + "x".join([str(x) for x in resize_shape]) + ".npy"
    npy_file = parent_path.joinpath(fname)
    if npy_file.exists():
        print("loading from file")
       
        saved_frames = np.load(npy_file)
        return saved_frames
            
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
    saved_frames = np.array(frames)
   
    np.save(npy_file, saved_frames)

    return saved_frames

"""
Helper method to load the given dataset into memory
"""

def load_dataset(dataset_root, json_filepath, file_type,label_index_file, resize=False, resize_shape=(112, 112)):
    

    def load_dataset_gen():
        print(pathlib.Path.cwd())
        ds_root = pathlib.Path(str(dataset_root))
        print(ds_root, ds_root.exists())
        if not ds_root.exists():
            raise ValueError("Dataset root folder does not exist: {0}".format(dataset_root))
        json_path = ds_root.joinpath(json_filepath)
        
        if not json_path.exists():
            raise ValueError("Jsonfile {0} does not exist".format(json_filepath))   
        with open(json_path) as f:
            dataset_files = json.load(f)['tubes']
        classes_filepath = ds_root.joinpath(label_index_file)
        if not classes_filepath.exists():
            raise ValueError("Label index file does not exist: {0}".format(label_index_file))
        
    
        class_labels_map = dict()
        with open(classes_filepath,'r+') as f:
            class_labels_map = {v:k for k,v in enumerate([l[:-1] for l in f.readlines()])}
        for video in dataset_files:
            path = ds_root.joinpath( "videos", file_type, video['path'])
            y = np.zeros(len(class_labels_map), dtype=np.uint8)
            for c in video['label']:
                y[class_labels_map[c]] = 1
            yield get_video_frames(str(path)), y
        
        return 
    return load_dataset_gen
