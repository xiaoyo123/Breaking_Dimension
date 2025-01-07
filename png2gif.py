from PIL import Image
import glob
import re
import cv2
import numpy as np

# 讀取所有PNG文件
frames = []
imgs = glob.glob("image/*.png")
for i in range(1, 16):
    new_frame = Image.open("image/" + str(i) + ".png")
    frames.append(new_frame)

for i in range(15, 0, -1):
    new_frame = Image.open("image/" + str(i) + ".png")
    frames.append(new_frame)

# 保存為GIF
frames[0].save('results/animated.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=100, loop=0, disposal=2)
