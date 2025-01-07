import cv2
from PIL import Image, ImageEnhance
import numpy as np

def resize_image(input_image_path, output_image_path, target_width, target_height):#size 
    img = cv2.imread(input_image_path)
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_image_path, resized_img)

def light(input_image_path, output_image_path, factor):
    # 打開輸入圖片
    image = Image.open(input_image_path)
    
    # 創建一個亮度增強對象
    enhancer = ImageEnhance.Brightness(image)
    
    # 增加圖片亮度
    image_light = enhancer.enhance(factor)
    
    # 保存輸出的圖片
    image_light.save(output_image_path)


def gray(input_image_path, output_image_path, desaturation_factor):
    # 打開輸入圖片並轉換為RGB格式
    image = Image.open(input_image_path).convert("RGB")
    image = np.array(image)  # 將PIL圖像轉換為numpy數組
    
    # 將圖片從RGB轉換為HSV
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # 減少飽和度
    img_hsv[:, :, 1] *= (1 - desaturation_factor)
    
    # 確保飽和度在0到255之間
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
    
    # 將HSV轉換回RGB
    img_hsv = img_hsv.astype(np.uint8)
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite(output_image_path, img_rgb)

input_image = 'results/photo.png'
output_image = 'results/photo.png'
target_width = 427
target_height = 1455

#gray(input_image, output_image, 0.9)
#light(output_image, output_image, 2.5)
resize_image(input_image, output_image, target_width, target_height)
