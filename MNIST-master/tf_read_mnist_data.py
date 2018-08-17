
# 本案例 是读取MNIST手写数字集的数字, 并转化成csv格式的数据,并使用Matplotlib绘制出来

import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)


    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


fig, ax = plt.subplots(
    nrows=5,
    ncols=5,
    sharex=True,
    sharey=True, )
ax = ax.flatten()
X_train, y_train = load_mnist('./MNIST_data')
X_test, y_test = load_mnist('./MNIST_data', kind='t10k')

# 使用csv存储和加载images和labels数据
def save_into_csv(X_train, y_train):
    np.savetxt('./train_img.csv', X_train,
           fmt='%i', delimiter=',')
    np.savetxt('train_labels.csv', y_train,
           fmt='%i', delimiter=',')
    X_train = np.genfromtxt('train_img.csv',
                        dtype=int, delimiter=',')
    y_train = np.genfromtxt('train_labels.csv',
                        dtype=int, delimiter=',')

np.savetxt('./1_test_img.csv', X_test)
np.savetxt('./_test_labels.csv', y_test)

# for i in range(5):
#     img = X_train[y_train == i][0].reshape(28, 28)
#     ax[i].imshow(img, cmap='Greys', interpolation='nearest')

for i in range(25):
    img = X_train[y_train == 1][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()