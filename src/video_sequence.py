import keras
import numpy as np
import pathlib
from video_utils import get_video_frames
import json
class VideoframeGenerator(keras.utils.Sequence):
    def __init__(self,dataset_root, json_filename,dataset_type,label_file, resize=True, resize_shape=(112, 112), batch_size=1, frames=128, shuffle=True):
        super(VideoframeGenerator, self).__init__()
        self.dataset_root = dataset_root
        self.json_filename = json_filename
        self.dataset_type = dataset_type
        self.resize = resize
        self.resize_shape = resize_shape
        self.batch_size = batch_size
        self.basepath = pathlib.Path(dataset_root)
        self.frames = frames
        self.shuffle = shuffle
        if not self.basepath.exists():
            raise ValueError("Invalid basepath {0}".format(dataset_root))
        v_path = self.basepath.joinpath(json_filename)
        if not v_path.exists():
            raise ValueError("No jsonfile in basebath named {0}".format(json_filename))
        with open(str(v_path), "r+") as f:
            self._videos = json.load(f)['tubes' ]
        label_file_path = self.basepath.joinpath(label_file)
        if not label_file_path.exists():
            raise ValueError("Not label file with name {0} found in basepath".format(label_file))
        self.class_labels_map = dict()
        with open(str(label_file_path),'r+') as f:
            self.class_labels_map = {v:k for k,v in enumerate([l[:-1] for l in f.readlines()])}
        self.on_epoch_end()
        
        
    def __len__(self):
        return int(np.floor(len(self._videos) / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self._videos))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
        list_id_temp = [self._videos[k] for k in indexes]
        X, y = self.__data_generation(list_id_temp)
        return X, y

    def __data_generation(self, dataset_files):
        X = np.empty((self.batch_size,self.frames, *self.resize_shape, 3))
        Y = np.empty((self.batch_size, 26))
        for i, video in enumerate(dataset_files):
            path = self.basepath.joinpath( "videos", self.dataset_type, video['path'])
            y = np.zeros(len(self.class_labels_map), dtype=np.uint8)
            for c in video['label']:
                y[self.class_labels_map[c]] = 1
            Y[i, ] = y
            x = get_video_frames(str(path), self.resize, self.resize_shape)
            if x.shape[0] < self.frames:
                x = np.pad(x, ((0, self.frames - x.shape[0]), (0, 0), (0, 0), (0, 0)), 'constant')
            X[i,] = x/255.0
        
        return X, Y
    
    
