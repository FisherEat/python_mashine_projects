'''

本案例用来测试线性模型 ,使用mashine_d_1.csv中的数据

'''

# import numpy as np
# import matplotlib.pyplot as plt
#
# def createdata(filename):
#     file = open(filename)
#     lines = file.readlines()
#     rows = len(lines)
#     dataMatrix = np.zeros((rows, 2))
#     print(lines)
#     print(dataMatrix)
#     row = 0
#     for line in lines:
#         line = line.strip().split('\t')
#         for word in line:
#             line = word.split()
#             line = list(map(float, line))
#             print(line)
#         dataMatrix[row, :] = line[:]
#         row += 1
#     return dataMatrix
#
# def drawplot(dataset):
#     fig = plt.figure(1)
#     ax = fig.add_subplot(111)
#     ax.scatter(dataset[:, 1], dataset[:, 0], c='r', marker='^')
#     plt.show()
#
# if __name__ == '__main__':
#     dataset = createdata('./mashine_d_1.csv')
#     drawplot(dataset)

#encoding=UTF-8
'''''

本案例用来测试线性模型,使用矩阵推算出线性模型的最优解


'''

# from numpy import *
# import matplotlib.pyplot as plt
# from random import *
#
# def loadData():
#     x = arange(-1,1,0.02)
#     y = ((x*x-1)**3+1)*(cos(x*2)+0.6*sin(x*1.3))
#     # 生成的曲线上的各个点偏移一下，并放入到xa,ya中去
#     xr = []
#     yr = []
#     i = 0
#     for xx in x:
#         yy = y[i]
#         d = float(randint(80, 120))/100
#         i += 1
#         xr.append(xx*d)
#         yr.append(yy*d)
#     return x, y, xr, yr
#
# def XY(x, y, order):
#     X = []
#     for i in range(order+1):
#         X.append(x**i)
#     X = mat(X).T
#     Y = array(y).reshape((len(y), 1))
#     return X, Y
#
# def figPlot(x1, y1, x2, y2):
#     plt.plot(x1, y1, color='g', linestyle='-', marker='')
#     plt.plot(x2, y2, color='m', linestyle='', marker='.')
#     plt.show()
#
# def Main():
#     x, y, xr, yr = loadData()
#     X, Y = XY(x, y, 9)
#     XT = X.transpose()#X的转置
#     B = dot(dot(linalg.inv(dot(XT, X)), XT), Y)#套用最小二乘法公式
#     myY = dot(X, B)
#     figPlot(x, myY, xr, yr)
#
# Main()


'''
本例用线性模型计算出转化率之间的线性关系
'''
# encoding=UTF-8
import numpy as np
import matplotlib.pyplot as plt

def createdata(filename):
    file = open(filename)
    lines = file.readlines()
    rows = len(lines)
    dataMatrix = np.zeros((rows, 2))
    print(lines)
    print(dataMatrix)
    dataSet = []
    row = 0
    for line in lines:
        line = line.strip().split('\t')
        for word in line:
            line = word.split()
            line = list(map(float, line))
            print(line)
        dataMatrix[row, :] = line[:]
        row += 1

    return dataMatrix

def drawplot(dataset):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.scatter(dataset[:, 1], dataset[:, 0], c='r', marker='^')
    plt.show()

if __name__ == '__main__':
    dataset = createdata('./cat_d_1.txt')
    drawplot(dataset)

'''
本例 用一个更加简洁的方式将txt中的数据转换成二维数组
'''
# encoding=UTF-8
import numpy as np
import matplotlib.pyplot as plt

def createdata(filename):
    dataSet = []
    fileIn = open(filename)
    for line in fileIn.readlines():
        lineArr = line.strip().split('\t')
        print(lineArr)

        dataSet.append([float(lineArr[0]), float(lineArr[1])])
    # file = open(filename)
    # lines = file.readlines()
    # rows = len(lines)
    # # dataMatrix = np.zeros((rows, 2))
    # # print(lines)
    # # print(dataMatrix)
    # dataSet = []
    # for line in lines:
    #     lineArray = line.strip().split('\t')
    #     dataSet.append([float(lineArray[0]), float(lineArray)[1]])
    # # row = 0
    # for line in lines:
    #     line = line.strip().split('\t')
    #     for word in line:
    #         line = word.split()
    #         line = list(map(float, line))
    #         print(line)
    #     dataMatrix[row, :] = line[:]
    #     row += 1

    print(dataSet)
    dataMatrix = np.mat(dataSet)
    print(dataMatrix)
    return dataMatrix

def drawplot(dataset):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    numSamples, dim = dataset.shape
    x = []
    y = []
    for i in range(numSamples):
        x.append(dataset[i, 0])
        y.append(dataset[i, 1])
    # ax.scatter(dataset[:, 1], dataset[:, 0], c='r', marker='^')
    ax.scatter(x, y, c='r', marker='^')
    plt.show()

def drawDS(dataSet):
    numSamples, dim = dataSet.shape
    # draw all samples
    for i in range(numSamples):
        plt.plot(dataSet[i, 0], dataSet[i, 1], c='r', marker='^')

    plt.show()

if __name__ == '__main__':
    dataset = createdata('./mashine_d_4.txt')

    # drawDS(dataset)
    drawplot(dataset)
