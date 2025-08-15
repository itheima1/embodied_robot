import matplotlib.pyplot as plt
import pylab
import cv2
import numpy as np
img = plt.imread("a.jpg")
plt.imshow(img) #显示原图
pylab.show()
fil = np.array([[ 0,1,0], #卷积核
                [ 1,-4,1],
                [ 0,1,0]])
res = cv2.filter2D(img,-1,fil) #使用opencv的卷积函数
plt.imshow(res) #显示卷积后的图片
pylab.show()