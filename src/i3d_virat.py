from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Flatten
from i3d_inception import Inception_Inflated3d
from video_sequence import VideoframeGenerator
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
    dataset_root = "/workspaces/i3d_keras/dataset/TinyVIRAT/"
    shape = (256, 112, 112, 3)
    training_generator = VideoframeGenerator(dataset_root, "tiny_train.json","train", "classes.txt")
    test_generator = VideoframeGenerator(dataset_root, "tiny_test.json","test", "classes.txt")
    model = get_i3d_virat((256, 112, 112, 3), 'rgb_kinetics_only')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit_generator(training_generator, epochs=1)
    model.evaluate_generator(test_generator)
if __name__ == '__main__':
    main()
