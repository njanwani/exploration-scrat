"""
This file generates a video from the generated images for an exploration
run
"""

import cv2
import time
import glob
IMAGE_DIR = r'/Users/neiljanwani/Documents/exploration-scrat/images/map20_eig'
SAVE_DIR = r'/Users/neiljanwani/Documents/exploration-scrat/images/final'

img_array = []
for filename in glob.glob(IMAGE_DIR + '/*.png'):
    try:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    except:
        print(f'Lost image {filename}')
 
 
out = cv2.VideoWriter(f'{SAVE_DIR}/exploration_{str(time.asctime()).replace(" ","_").replace(":","-")}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()