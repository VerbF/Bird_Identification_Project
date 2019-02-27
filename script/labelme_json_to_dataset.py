# -*- coding: utf-8 -*-
import os

i = 7
for i in range(7,11):
    path = 'E:/TensorFlowTools/Bird_Identification_Project/dataset/points_' + str(i) + '/pic_json/'
    json_file = os.listdir(path)
    for file in json_file:
        os.system("python E:/Anaconda/envs/labelme/Scripts/labelme_json_to_dataset.exe %s"%(path + file))
    