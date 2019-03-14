# Working code. Writes to the frames directory with format %04d.jpg

import cv2
import os

path = '/home/jerryant/Desktop/Walking/test-9/gaitanalysis/'
name = '/Cycling_for_Freezing_Gait_in_Parkinson_Disease.mp4'
os.chdir(path)

vidcap = cv2.VideoCapture(path+name)
success, image = vidcap.read()
count = 0
success = True
while success:
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    cv2.imwrite("frames/%04d.jpg" % count, image)
    count += 1
