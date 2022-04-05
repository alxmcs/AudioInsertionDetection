from matplotlib import pyplot as plt
import numpy as np


def display_results(same_pairs_distance, different_pairs_distance):
    """
    displays a plot with two distance sequences
    :param same_pairs_distance: a 1-d list with euclidean distances
    between feature vectors obtained from the samples taken from the same audio
    :param different_pairs_distance: a 1-d list with euclidean distances
    between feature vectors obtained from the samples taken from the different audios
    :return:  window with 1-d plot
    """
    plt.figure(figsize=(30, 2))
    diff_y = np.ones(len(same_pairs_distance))
    same_y = np.ones(len(different_pairs_distance))
    plt.yticks([])
    plt.bar(same_pairs_distance, same_y / 2, color='g', width=0.005)
    plt.bar(different_pairs_distance, diff_y, color='r', width=0.0025)
    plt.show()

if __name__== "__main__":
    same = np.random.ranf(10)
    print(type(same))
    print(same)
    different = np.random.ranf(10)
    display_results(same, different)
