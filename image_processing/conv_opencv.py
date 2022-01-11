import cv2 as cv
import numpy as np

img = cv.imread("./data/custom_cat_dog/cat1.jpg")

# kernel = np.ones((5, 5), np.float32) / 25
kernel = np.ones((5, 5), np.float32) / 25
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])


kernel1 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
kernel2 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
kernel3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

# print(kernel.shape)
filtered_img1 = cv.filter2D(img, -1, kernel1)
filtered_img2 = cv.filter2D(img, -1, kernel2)
filtered_img3 = cv.filter2D(img, -1, kernel3)

filtered_img4 = cv.filter2D(img, -1, emboss_kernel)


# cv.imshow("1", filtered_img1)
# cv.imshow("2", filtered_img2)
# cv.imshow("3", filtered_img3)
cv.imshow("4", filtered_img4)

# print(img[:, :, 0].shape)
img_r = img[:, :, 0]
print(np.shape(img_r))

print(np.max(img_r))

m, n = 3, 4
k = 3
for i in range(m):
    for j in range(n):
        img_r[:]


# cv.imshow("R", img[:, :, 2])
cv.waitKey()
