from keras.models import load_model
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np
from video_sequence import VideoframeGenerator
def evaluate_model(model_filepath, test_dataset):
    model = load_model(model_filepath)
    print("model loaded")
    predictions = list()
    trues = list()
    count = 0
    for batch, labels in test_dataset:
        v = model.predict(batch)
        for y_t, y_p in zip(labels, v):
            p = np.array([1 if z >=0.5 else 0 for z in y_p])
            predictions.append(p)
            trues.append(y_t)
        count+=1
        print(count)
    f1_macro = f1_score(trues, predictions, average='macro')
    f1_micro = f1_score(trues, predictions, average='micro')
    accuracy = accuracy_score(trues, predictions)
    print(f1_macro, f1_micro, accuracy)
    return f1_macro, f1_micro, accuracy
def iterate_over_sequence(sequence):
    for c in sequence:
        print(len(c))


def main():
    model_path = '/workspaces/i3d_keras/prp/i3d_bicubic_v1.h5'
    dataset_root = "/workspaces/i3d_keras/dataset/TinyVIRAT/"
    test_generator = VideoframeGenerator(dataset_root, "tiny_test.json","test", "classes.txt", frames=256, batch_size=5, shuffle=False)
    training_generator = VideoframeGenerator(dataset_root, "tiny_train.json","train", "classes.txt", frames=256, batch_size=5, shuffle=False)
    iterate_over_sequence(training_generator)


if __name__ == '__main__':
    main()

