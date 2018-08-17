'''
本案例用来加强python字符串学习
'''

'''
本案例采用特定格式的数据,然后对数据进行处理
'''
def readData(fileName):
    fileIn = open(fileName, encoding='utf8')
    x_dataSet = []
    y_dataSet = []
    for line in fileIn.readlines():
        print(line)
        lineArr = line.strip().split(':')
        print(lineArr)
        x_dataSet.append(lineArr[0].strip("'"))
        y_dataSet.append(lineArr[1].strip("'"))

    print("x_data:", x_dataSet)
    print("y_data:", y_dataSet)

# readData('./event_data.txt')

'''
字符串常用操作
'''


