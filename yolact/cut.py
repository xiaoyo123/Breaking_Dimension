from PIL import Image

def crop_image(image_path, percentage):
    # 打開圖片
    image = Image.open(image_path)
    width, height = image.size
    
    # 計算新的高度
    new_height = int(height * (percentage / 100))
    
    # 裁剪圖片
    cropped_image = image.crop((0, 0, width, new_height))
    
    # 保存裁剪後的圖片
    cropped_image.save("result/cutphoto.png")
    cropped_image.show()  # 打開顯示圖片

# 使用範例：輸入 90 裁剪圖片的上90%
image_path = "result/photo.png"  # 請將此處替換為你的圖片路徑
percentage = 70  # 輸入你想保留的百分比
crop_image(image_path, percentage)
