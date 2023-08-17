import cv2
import os

dataPath = os.path.dirname(os.path.realpath(__file__)) + "\\misc\\"
inferenceImage = dataPath + "car.jpg"

imageorg = cv2.imread(inferenceImage, -1)
cv2.imshow("1", imageorg)
cv2.waitKey()