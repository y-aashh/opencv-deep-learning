
"""
preworked sample
"""

import numpy as np 
import cv2

labels = ["dog","cat"]
np.random.seed(1)

W = np.random.randn(3,3072)
b = np.random.randn(3)

img = cv2.imread("dog.jpg")

image = cv2.resize(img,(32,32)).flatten()  #flatten => 32*32*3 to 3072*1

scores = W.dot(image) + b

cv2.putText(img, "Label: {}".format(labels[np.argmax(scores)]),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("guess1",img)
cv2.waitKey(0)