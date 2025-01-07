import cv2

image = cv2.imread("results/photo.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

coords = cv2.findNonZero(gray)
x, y, w, h = cv2.boundingRect(coords)

image = image[y:y+h, x:x+w]

h = image.shape[0]
w = image.shape[1]

for x in range(w):
    for y in range(h):
        if gray[y, x] == 0:
            image[y, x, 3] = 0

cv2.imwrite('results/photo.png', image)

