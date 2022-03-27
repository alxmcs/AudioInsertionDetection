import librosa
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras import models
from keras.layers import Input
from audio_preprocessing import preprocess_audio, METHODS
from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3

MODELS = ['ResNet', 'DenseNet', 'Inception']
SHAPE = (128, 128, 3)


class FeatureExtractionModel:
    """
    a sample class to utilize pretrained fine-tuned tf models
    """
    def __init__(self, name=MODELS[0], weights='imagenet'):
        """
        initializes a model
        :param name: name of the model
        :param weights: a path to its weights
        """
        if name == MODELS[0]:
            base_model = ResNet50(weights=weights, input_tensor=Input(shape=SHAPE), include_top=False)
        elif name == MODELS[1]:
            base_model = DenseNet121(weights=weights, input_tensor=Input(shape=SHAPE), include_top=False)
        elif name == MODELS[2]:
            base_model = InceptionV3(weights=weights, input_tensor=Input(shape=SHAPE), include_top=False)
        else:
            raise ValueError(
                f"{datetime.now()}: Unsupported model. Supported models are {MODELS}'.")
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        final_model = models.Sequential()
        final_model.add(base_model)
        final_model.add(global_average_layer)
        self.model = final_model

    @staticmethod
    def __validate_data(data):
        if len(data.shape) == 4 and data.shape[1] == 128 and data.shape[2] == 128 and data.shape[3] == 3:
            return True
        else:
            return False

    def get_feature_vectors(self, data):
        if self.__validate_data(data):
            return self.model.predict(data)


if __name__ == "__main__":
    # these two lines are needed to run inference on my dated gtx 1660 ti
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    ##tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # the following is an example of extracting a feature vector from a single 128*128 spectrogram piece
    parser = argparse.ArgumentParser(description="feature vector extracting script")
    parser.add_argument("-m", dest="model", type=str, choices=MODELS, help="name of the model to use")
    parser.add_argument("-w", dest="weights", type=str, default='imagenet', help="path to the model weights")
    parser.add_argument("-a", dest="audio", type=str, help="path to an audio file")
    parser.add_argument("-p", dest="preproc", type=str, choices=METHODS, default='palanisamy', help="name of the "
                                                                                                    "preprocessing "
                                                                                                    "method")
    args = parser.parse_args()
    print(args)
    if args.model:
        fem = FeatureExtractionModel(args.model, args.weights)
    else:
        fem = FeatureExtractionModel(weights=args.weights)

    if args.audio:
        a, sr = librosa.load(args.audio)
    else:
        a, sr = librosa.load(r'C:\Users\stron\Documents\GitHub\docs\musicDS\170439__argande102__wind-on-microphone.mp3')
        print(librosa.ex('trumpet'))

    preproc_audio = tf.expand_dims(preprocess_audio(a, sr, args.preproc)[:, 0:128, :], axis=0)
    vector = fem.get_feature_vectors(preproc_audio)
    print(f'{datetime.now()}: got {vector.shape[1]}-dimensional vector from {args.model if args.model else MODELS[0]}')
    print(np.linalg.norm(vector-vector))