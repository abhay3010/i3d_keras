from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Flatten
from i3d_inception import Inception_Inflated3d
from keras.callbacks import ModelCheckpoint
from video_sequence import VideoframeGenerator
from keras.models import load_model

import pathlib
def get_i3d_virat(input_shape,weights=None, dropout_prob=0.0):
    base_layer = Sequential()
    prev_model = Inception_Inflated3d(False,input_shape=input_shape, endpoint_logit=True, dropout_prob=dropout_prob)
    print(prev_model.summary())
    base_layer.add(prev_model)
    base_layer.add(Flatten())
    base_layer.add(Dense(26, activation='sigmoid'))
    print(base_layer.summary())
    return base_layer

def main():
    #dataset_root = "/virat-vr/TinyVIRAT/"
    checkpoint_path = "/virat-vr/tmp1.hdf5"
    checkpoint_path = ""

    dataset_root = "/workspaces/i3d_keras/dataset/TinyVIRAT"
    shape = (256, 112, 112, 3)
    training_generator = VideoframeGenerator(dataset_root, "tiny_train.json","train", "classes.txt")
    test_generator = VideoframeGenerator(dataset_root, "tiny_test.json","test", "classes.txt")
    model = None
    if pathlib.Path(checkpoint_path).exists():
        model = load_model(checkpoint_path)
    else:    
        model = get_i3d_virat(shape, 'rgb_kinetics_only')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=False)

    model.fit_generator(training_generator, epochs=1, use_multiprocessing=True,workers=4,max_queue_size=10, callbacks=[checkpointer] )
    #model.evaluate_generator(test_generator)
if __name__ == '__main__':
    main()
