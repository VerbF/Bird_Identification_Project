import os
import sys
import shutil
sourceDir = 'E:/TensorFlowTools/Bird_Identification_Project/dataset/points_6/dataset_json/'
targetDir = 'E:/TensorFlowTools/Bird_Identification_Project/dataset/points_6/cv2_mask/'
dirList = os.listdir(sourceDir) #列出文件夹下所有的目录与文件
for i in range(0,len(dirList)):
    sourceName = sourceDir + dirList[i] + '/label.png'
    newImgName = dirList[i][0:9] + '.png'
    targetName = targetDir + newImgName
    shutil.copyfile(sourceName,targetName)
    print(newImgName + ' OK !')
