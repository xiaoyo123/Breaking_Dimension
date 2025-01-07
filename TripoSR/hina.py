import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def crop_white_background(image_path, output_path):
    # 讀取圖片
    image = cv2.imread(image_path)
    
    # 將圖片轉為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 將灰階圖片進行二值化處理，生成二值掩碼
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # 反轉二值掩碼
    thresh = cv2.bitwise_not(thresh)
    
    # 在二值掩碼中找到輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 獲取最大的輪廓的邊界矩形
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # 使用邊界矩形裁剪圖片
        cropped_image = image[y:y+h, x:x+w]
        
        # 保存裁剪後的圖片
        cv2.imwrite(output_path, cropped_image)
    else:
        print(f"未找到任何輪廓：{image_path}")

def process_folder(input_folder, output_folder):
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍歷輸入資料夾中的所有檔案
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # 檢查是否為圖片檔案（可根據需要擴展支持的格式）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            crop_white_background(input_path, output_path)
        else:
            print(f"忽略非圖片檔案：{filename}")

# 使用範例
input_folder = 'images/name'
output_folder = 'images/blueArchive'
process_folder(input_folder, output_folder)

"""
程式碼說明：

讀取圖片：使用 cv2.imread() 函數讀取輸入圖片。

轉為灰階：使用 cv2.cvtColor() 將圖片轉為灰階，因為在灰階圖片中處理二值化會更簡單。

二值化處理：使用 cv2.threshold() 將灰階圖片轉換為二值圖像，其中白色部分將被轉換為黑色，背景將成為白色。

反轉掩膜：使用 cv2.bitwise_not() 反轉二值圖像，這樣物品部分將變為白色，而背景將變為黑色。

找到輪廓：使用 cv2.findContours() 在反轉後的二值圖像中找到所有外部輪廓。

獲取邊界矩形：使用 cv2.boundingRect() 獲取最大的輪廓的邊界矩形。

裁剪圖片：使用邊界矩形的坐標裁剪原始圖片，得到去除背景後的圖像。
"""