import numpy as np
import matplotlib.pyplot as plt


from PIL import Image


def get_t(t):
    return np.array([t**2, t, 1])


curve1 = np.array([[0.65, 0.65, 0],
      [0, 0.65, 0.65]]) * 512

curve2 = np.array([[0, 0.35, 0.35],
       [0.35, 0.35, 0]]) * 512
#curve = np.array([[0.65,0.65,0.65],
#                  [0, 0.5, 1]]) * 512
m = np.array([[1, -2, 1],
              [-2, 2, 0],
              [1,  0, 0]])
img = Image.open("./104_road textures pach-seamless/road texture pack-seamless (8).jpg")
img = img.rotate(270, Image.NEAREST, expand = 1)
img = np.array(img)


plt.imshow(img)
for i in range(3):
    plt.plot(curve1[0,i], 512 - curve1[1,i], "wo")
    plt.plot(curve2[0,i], 512 - curve2[1,i], "wo")
for t in np.arange(0, 1.1, 0.1):
    point = np.matmul(np.matmul(curve1, m), get_t(t))
    plt.plot(point[0], 512 - point[1], "ro")
    point = np.matmul(np.matmul(curve2, m), get_t(t))
    plt.plot(point[0], 512 - point[1], "bo")
plt.show()