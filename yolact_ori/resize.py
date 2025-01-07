import cv2

def resize_image(input_image_path, output_image_path, target_width, target_height):
    img = cv2.imread(input_image_path)
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_image_path, resized_img)

# 使用範例
input_image = 'result/3.png'
output_image = 'result/3re.png'
target_width = 427
target_height = 1455

resize_image(input_image, output_image, target_width, target_height)
