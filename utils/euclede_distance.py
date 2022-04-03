import numpy as np
from feature_vector_extracting import vector_gener
def euclede_calc():
    arr_1st_method_1 = vector_gener(r'C:\Users\stron\Documents\GitHub\docs\musicDS\158340_bpt10zbu_fr-157.ogg', 'guzhov')
    arr_1st_method_2 = vector_gener(r'C:\Users\stron\Documents\GitHub\docs\musicDS\170439_argande102_wind-on-microphone.ogg', 'guzhov')
    arr_2nd_method_1 = vector_gener(r'C:\Users\stron\Documents\GitHub\docs\musicDS\158340_bpt10zbu_fr-157.ogg', False)
    arr_2nd_method_2 = vector_gener(r'C:\Users\stron\Documents\GitHub\docs\musicDS\170439_argande102_wind-on-microphone.ogg', False)
    res_arr1 = []
    res_arr2 = []
    res_arr1.append(np.linalg.norm(arr_1st_method_1[0],arr_1st_method_1[1]))
    res_arr1.append(np.linalg.norm(arr_1st_method_1[1],arr_1st_method_1[2]))
    res_arr1.append(np.linalg.norm(arr_1st_method_1[2],arr_1st_method_1[3]))
    res_arr1.append(np.linalg.norm(arr_1st_method_1[3],arr_1st_method_1[4]))
    res_arr1.append(np.linalg.norm(arr_1st_method_1[4],arr_1st_method_1[5]))
    res_arr1.append(np.linalg.norm(arr_1st_method_1[5],arr_1st_method_1[6]))
    res_arr1.append(np.linalg.norm(arr_1st_method_1[6],arr_1st_method_1[7]))
    res_arr1.append(np.linalg.norm(arr_1st_method_1[7],arr_1st_method_1[8]))
    res_arr1.append(np.linalg.norm(arr_1st_method_1[8],arr_1st_method_1[9]))
    res_arr1.append(np.linalg.norm(arr_1st_method_1[9],arr_1st_method_1[10]))
    res_arr2.append(np.linalg.norm(arr_1st_method_2[0],arr_1st_method_2[1]))
    res_arr2.append(np.linalg.norm(arr_1st_method_2[1],arr_1st_method_2[2]))
    res_arr2.append(np.linalg.norm(arr_1st_method_2[2],arr_1st_method_2[3]))
    res_arr2.append(np.linalg.norm(arr_1st_method_2[3],arr_1st_method_2[4]))
    res_arr2.append(np.linalg.norm(arr_1st_method_2[4],arr_1st_method_2[5]))
    res_arr2.append(np.linalg.norm(arr_1st_method_2[5],arr_1st_method_2[6]))
    res_arr2.append(np.linalg.norm(arr_1st_method_2[6],arr_1st_method_2[7]))
    res_arr2.append(np.linalg.norm(arr_1st_method_2[7],arr_1st_method_2[8]))
    res_arr2.append(np.linalg.norm(arr_1st_method_2[8],arr_1st_method_2[9]))
    res_arr2.append(np.linalg.norm(arr_1st_method_2[9],arr_1st_method_2[10]))
    res_arr1.append(np.linalg.norm(arr_2nd_method_1[0],arr_2nd_method_1[1]))
    res_arr1.append(np.linalg.norm(arr_2nd_method_1[1],arr_2nd_method_1[2]))
    res_arr1.append(np.linalg.norm(arr_2nd_method_1[2],arr_2nd_method_1[3]))
    res_arr1.append(np.linalg.norm(arr_2nd_method_1[3],arr_2nd_method_1[4]))
    res_arr1.append(np.linalg.norm(arr_2nd_method_1[4],arr_2nd_method_1[5]))
    res_arr1.append(np.linalg.norm(arr_2nd_method_1[5],arr_2nd_method_1[6]))
    res_arr1.append(np.linalg.norm(arr_2nd_method_1[6],arr_2nd_method_1[7]))
    res_arr1.append(np.linalg.norm(arr_2nd_method_1[7],arr_2nd_method_1[8]))
    res_arr1.append(np.linalg.norm(arr_2nd_method_1[8],arr_2nd_method_1[9]))
    res_arr1.append(np.linalg.norm(arr_2nd_method_1[9],arr_2nd_method_1[10]))
    res_arr2.append(np.linalg.norm(arr_2nd_method_2[0],arr_2nd_method_2[1]))
    res_arr2.append(np.linalg.norm(arr_2nd_method_2[1],arr_2nd_method_2[2]))
    res_arr2.append(np.linalg.norm(arr_2nd_method_2[2],arr_2nd_method_2[3]))
    res_arr2.append(np.linalg.norm(arr_2nd_method_2[3],arr_2nd_method_2[4]))
    res_arr2.append(np.linalg.norm(arr_2nd_method_2[4],arr_2nd_method_2[5]))
    res_arr2.append(np.linalg.norm(arr_2nd_method_2[5],arr_2nd_method_2[6]))
    res_arr2.append(np.linalg.norm(arr_2nd_method_2[6],arr_2nd_method_2[7]))
    res_arr2.append(np.linalg.norm(arr_2nd_method_2[7],arr_2nd_method_2[8]))
    res_arr2.append(np.linalg.norm(arr_2nd_method_2[8],arr_2nd_method_2[9]))
    res_arr2.append(np.linalg.norm(arr_2nd_method_2[9],arr_2nd_method_2[10]))
    return