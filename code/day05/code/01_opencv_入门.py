import cv2 as cv
import numpy as np

# 构建一个空白的矩阵
img = np.zeros((200,200,3),np.uint8)

# 将第15行所有像素点全都改成红色
for i in range(200):
    # 设置第15行颜色为红色
    img[15,i] = (0,0,255)

# 显示图片
cv.imshow("src",img)

cv.waitKey(0)
cv.destroyAllWindows()
