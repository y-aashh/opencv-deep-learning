import cv2
import numpy as np 
import os

class SimplePreprocessor:
    def __init__ (self,width,height,inter = cv2.INTER_AREA):

        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self,image):
        return cv2.resize(image,(self.width,self.height),interpolation = self.inter)

class SimpleDatasetLoader:
    def __init__(self,preprocessors = None):
        self.preprocessors = preprocessors
        
        if self.preprocessors is None:
            self.preprocessors = []
            
    def load(self,imagePaths,verbose = -1):
        data =[]
        lables = []
        for(i,imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            lable = imagePath.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            lables.append(lable)

        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
          print("[INFO] processed {}/{}".format(i + 1,len(imagePaths)))

        return (np.array(data), np.array(lables))  
