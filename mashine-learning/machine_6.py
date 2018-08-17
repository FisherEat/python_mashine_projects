'''
本案例用来测试分类算法: 朴素贝叶斯算法\
参考网址: https://blog.csdn.net/Dream_angel_Z/article/details/46120867
该案例过于复杂
'''

from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

# 下面的函数是根据上面给出来的样本数据所创建出来的一个词库。
# 该函数会将loadDataSet里所有的词汇转变成一个数组,即词库
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 下面的函数功能是把单个样本映射到词库中去，
# 统计单个样本在词库中的出现情况，1表示出现过，0表示没有出现，函数如下：
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# 计算条件概率和类标签概率
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs) #计算某个类发生的概率
    p0Num = ones(numWords);
    p1Num = ones(numWords) #初始样本个数为1，防止条件概率为0，影响结果
    p0Denom = 2.0; p1Denom = 2.0  #作用同上
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)         #计算类标签为1时的其它属性发生的条件概率
    p0Vect = log(p0Num/p0Denom)         #计算标签为0时的其它属性发生的条件概率
    return p0Vect, p1Vect, pAbusive       #返回条件概率和类标签为1的概率

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    # step1：加载数据集和类标号
    listOPosts, listClasses = loadDataSet()
    # step2：创建词库
    myVocabList = createVocabList(listOPosts)
    # step3：计算每个样本在词库中的出现情况
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # step4：调用第四步函数，计算条件概率
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    # step5
    # 测试1
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    # 测试2
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    # 测试3
    testEntry = ['dog', 'is']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

# testingNB()

'''
本案例用来测试分类算法: 朴素贝叶斯 病人分类算法,原书案例
参考网址: https://www.oudahe.com/p/46227/
'''
class NBClassify(object):
  def __init__(self, fillNa = 1):
    self.fillNa = 1
    pass
  def train(self, trainSet):
    # 计算每种类别的概率
    # 保存所有tag的所有种类，及它们出现的频次
    dictTag = {}
    for subTuple in trainSet:
      dictTag[str(subTuple[1])] = 1 if str(subTuple[1]) not in dictTag.keys() else dictTag[str(subTuple[1])] + 1
    # 保存每个tag本身的概率
    tagProbablity = {}
    totalFreq = sum([value for value in dictTag.values()])
    for key, value in dictTag.items():
      tagProbablity[key] = value / totalFreq
    # print(tagProbablity)
    self.tagProbablity = tagProbablity
    ##############################################################################
    # 计算特征的条件概率
    # 保存特征属性基本信息{特征1:{值1:出现5次, 值2:出现1次}, 特征2:{值1:出现1次, 值2:出现5次}}
    dictFeaturesBase = {}
    for subTuple in trainSet:
      for key, value in subTuple[0].items():
        if key not in dictFeaturesBase.keys():
          dictFeaturesBase[key] = {value:1}
        else:
          if value not in dictFeaturesBase[key].keys():
            dictFeaturesBase[key][value] = 1
          else:
            dictFeaturesBase[key][value] += 1
    # dictFeaturesBase = {
      # '职业': {'农夫': 1, '教师': 2, '建筑工人': 2, '护士': 1},
      # '症状': {'打喷嚏': 3, '头痛': 3}
      # }
    dictFeatures = {}.fromkeys([key for key in dictTag])
    for key in dictFeatures.keys():
      dictFeatures[key] = {}.fromkeys([key for key in dictFeaturesBase])
    for key, value in dictFeatures.items():
      for subkey in value.keys():
        value[subkey] = {}.fromkeys([x for x in dictFeaturesBase[subkey].keys()])
    # dictFeatures = {
    #  '感冒 ': {'症状': {'打喷嚏': None, '头痛': None}, '职业': {'护士': None, '农夫': None, '建筑工人': None, '教师': None}},
    #  '脑震荡': {'症状': {'打喷嚏': None, '头痛': None}, '职业': {'护士': None, '农夫': None, '建筑工人': None, '教师': None}},
    #  '过敏 ': {'症状': {'打喷嚏': None, '头痛': None}, '职业': {'护士': None, '农夫': None, '建筑工人': None, '教师': None}}
    #  }
    # initialise dictFeatures
    for subTuple in trainSet:
      for key, value in subTuple[0].items():
        dictFeatures[subTuple[1]][key][value] = 1 if dictFeatures[subTuple[1]][key][value] == None else dictFeatures[subTuple[1]][key][value] + 1
    # print(dictFeatures)
    # 将驯良样本中没有的项目，由None改为一个非常小的数值，表示其概率极小而并非是零
    for tag, featuresDict in dictFeatures.items():
      for featureName, fetureValueDict in featuresDict.items():
        for featureKey, featureValues in fetureValueDict.items():
          if featureValues == None:
            fetureValueDict[featureKey] = 1
    # 由特征频率计算特征的条件概率P(feature|tag)
    for tag, featuresDict in dictFeatures.items():
      for featureName, fetureValueDict in featuresDict.items():
        totalCount = sum([x for x in fetureValueDict.values() if x != None])
        for featureKey, featureValues in fetureValueDict.items():
          fetureValueDict[featureKey] = featureValues/totalCount if featureValues != None else None
    self.featuresProbablity = dictFeatures
    ##############################################################################
  def classify(self, featureDict):
    resultDict = {}
    # 计算每个tag的条件概率
    for key, value in self.tagProbablity.items():
      iNumList = []
      for f, v in featureDict.items():
        if self.featuresProbablity[key][f][v]:
          iNumList.append(self.featuresProbablity[key][f][v])
      conditionPr = 1
      for iNum in iNumList:
        conditionPr *= iNum
      resultDict[key] = value * conditionPr
    # 对比每个tag的条件概率的大小
    resultList = sorted(resultDict.items(), key=lambda x:x[1], reverse=True)
    return resultList[0][0]


def baysTest():
    trainSet = [
    ({"症状":"打喷嚏", "职业":"护士"}, "感冒 "),
    ({"症状":"打喷嚏", "职业":"农夫"}, "过敏 "),
    ({"症状":"头痛", "职业":"建筑工人"}, "脑震荡"),
    ({"症状":"头痛", "职业":"建筑工人"}, "感冒 "),
    ({"症状":"打喷嚏", "职业":"教师"}, "感冒 "),
    ({"症状":"头痛", "职业":"教师"}, "脑震荡"),
    ]
    monitor = NBClassify()
    # trainSet is something like that [(featureDict, tag), ]
    monitor.train(trainSet)
    # 打喷嚏的建筑工人
    # 请问他患上感冒的概率有多大？
    result = monitor.classify({"症状":"打喷嚏", "职业":"建筑工人"})
    print(result)

# baysTest()


'''
本案例用来测试分类算法之 支持向量机SVM,并使用sklearn框架
参考网址sklearn 框架: http://sklearn.apachecn.org/cn/0.19.0/
'''
from sklearn import svm

def testSVM():
    # 客户年龄
    x = [[34], [33], [32], [31], [30], [30], [25], [23], [22], [18]]
    # 客户质量
    y = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    # 现在把训练数据和对应的分类放入分类器中进行训练
    # 这里使用rbf
    clf = svm.SVC(kernel='rbf').fit(x, y)

    # 预测年龄30的人的质量
    p = [[30]]
    print(clf.predict(p))

testSVM()

