import os
import sys
import shutil

rootDir = 'E:\\TensorFlowTools\\Bird_Identification_Project\\dataset\\dataset_train'

def listDir(rootDir):
    for filename in os.listdir(rootDir):
        pathname = os.path.join(rootDir, filename)
        if (os.path.isfile(pathname)):
            if(filename.find('bird') != -1) :
                numStr = filename[5:9]
                num = int(numStr)
                if not ( (num >= 1 )&( num <= 36)  | (num >= 61) &( num <= 96)):
                    os.remove(pathname)
                    print(filename)
                
        else:
            if(filename.find('bird') != -1) :
                numStr = filename[5:9]
                num = int(numStr)
                if not ( (num >= 1 )&( num <= 36)  | (num >= 61) &( num <= 96)):
                    shutil.rmtree(pathname)
                    print(filename)
                #os.removedirs(pathname)
            else:
                listDir(pathname)

listDir(rootDir)