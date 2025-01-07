import cv2
import numpy as np
from PIL import Image

H = "result/head/3.png"
B = "result/body/M.png"

head = Image.open(H)
body = Image.open(B)

scale_factor = max(168/head.width,345/head.height)
print(scale_factor)

# Resize head image
new_width = int(head.width * scale_factor)
new_height = int(head.height * scale_factor)
resized_head = head.resize((new_width, new_height))


# Convert head and body to NumPy arrays
head_numpy = np.array(resized_head)
body_numpy = np.array(body)

# Define center and dimensions of head and body
Hheight, Hwidth = head_numpy.shape[:2]
Bheight, Bwidth = body_numpy.shape[:2]
Hcenter_x = Hwidth // 2
Hcenter_y = Hheight // 2
"""
Bcenter_x = int(input("input the centerx:"))
Bcenter_y = int(input("input the centery:"))
"""
Bcenter_x = 368
Bcenter_y = 180
# Overlay head onto body

for i in range(Hheight):
    for j in range(Hwidth):
        if (0 <= Bcenter_x + j - Hcenter_x < Bwidth) and (0 <= Bcenter_y + i - Hcenter_y < Bheight) and (not np.array_equal(head_numpy[i, j], [0, 0, 0])):
            body_numpy[Bcenter_y + i - Hcenter_y, Bcenter_x + j - Hcenter_x] = head_numpy[i, j]

# Save result
cv2.imwrite("result/test.png", cv2.cvtColor(body_numpy, cv2.COLOR_RGB2BGR))
res = Image.open("result/test.png")
res.show()
