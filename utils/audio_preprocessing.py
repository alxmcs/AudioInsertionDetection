import librosa
import argparse
import numpy as np
from datetime import datetime
from skimage import transform
from skimage.io import imshow
from matplotlib import pyplot as plt

# window and hop parameters from https://arxiv.org/pdf/2007.11154.pdf
WINDOW_SIZES = [25, 50, 100]
HOP_SIZES = [10, 25, 50]
METHODS = ['guzhov', 'palanisamy']
# paper uses milliseconds, librosa uses STFT bins
recalculate_from_ms_to_bins = np.vectorize(lambda ms, sr: int(round(ms * sr / 1000)))


def calculate_melspectrogram(audio, sr, hop, window):
    """
    calculates melspectrogram of a given audio with a given parameters  and converts it to a log scale
    :param audio: audio-time series
    :param sr: its sample rate
    :param hop: number of samples between successive frames of STFT
    :param window:length of the STFT window
    :return: melspectrogram in log scale
    """
    delta = 1e-6
    logmel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=int(sr * 0.2), hop_length=hop, win_length=window)
    return np.log(logmel + delta)


def preprocess_audio(audio, sr, type):
    """
    converts audio-time series into three-channel log melspectrogram via specified algorithm (based on
    https://arxiv.org/pdf/2007.11154.pdf and https://arxiv.org/pdf/2004.07301.pdf )
    :param audio: audio-time series
    :param sr: its sample rate
    :param type: type of preprocessing algorithm, can be either 'guzhov' or 'palanisamy'
    :return: three-channel log melspectrogram
    """
    window_length = recalculate_from_ms_to_bins(WINDOW_SIZES, sr)
    hop_length = recalculate_from_ms_to_bins(HOP_SIZES, sr)
    print(sr)
    print(window_length)
    print(hop_length)
    ch0 = calculate_melspectrogram(audio, sr, hop_length[0], window_length[0])
    if type == METHODS[0]:  # from https://arxiv.org/pdf/2004.07301.pdf and https://arxiv.org/pdf/2007.11154.pdf
        return np.dstack((ch0, ch0, ch0))
    elif type == METHODS[1]:  # from https://arxiv.org/pdf/2007.11154.pdf
        ch1 = calculate_melspectrogram(audio, sr, hop_length[1], window_length[1])
        ch2 = calculate_melspectrogram(audio, sr, hop_length[2], window_length[2])
        ch1 = transform.resize(ch1, ch0.shape)
        ch2 = transform.resize(ch2, ch0.shape)
        return np.dstack((ch0, ch1, ch2))
    else:
        raise ValueError(
            f"{datetime.now()}: Unsupported type of audio preprocessing type. Supported types are {METHODS}.")


def display_data_portion(data):
    """
    displays 128*128 portion of preprocessed audio file
    :param data: three-channel log melspectrogram
    :return: window with the channel-wise display of the data portion
    """
    fig = plt.figure(figsize=(15, 5))
    fig.add_subplot(1, 3, 1)
    imshow(data[0:128, 0:128, 0])
    fig.add_subplot(1, 3, 2)
    imshow(data[0:128, 0:128, 1])
    fig.add_subplot(1, 3, 3)
    imshow(data[0:128, 0:128, 2])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="audio preprocessing script")
    parser.add_argument("-a", dest="audio", type=str, help="path to an audio file")
    parser.add_argument("-p", dest="preproc", type=str, choices=METHODS, default='palanisamy', help="name of the "
                                                                                                    "preprocessing "
                                                                                                    "method")
    args = parser.parse_args()
    if args.audio:
        a, sr = librosa.load(args.audio)
    else:
        a, sr = librosa.load(librosa.ex('trumpet'))
    res = preprocess_audio(a, sr, args.preproc)
    display_data_portion(res)
