import pandas as pd
import numpy as np
from sklearn import svm

print("准备数据")
testDf = pd.read_csv("../data/winequality-white.csv", sep=";")
testData = testDf.values
wineDf = pd.read_csv("../data/winequality-white.csv", sep=";")
wineData = wineDf.values

dataColName = wineDf.columns
rsColName = dataColName[-1]
ftColName = list(dataColName[:len(dataColName) - 1])

testFeature = testDf[ftColName].values
testResult = testDf[rsColName].values
wineFeature = wineDf[ftColName].values[:100]
wineResult = wineDf[rsColName].values[:100]

print("建立模型")
clf = svm.SVC(gamma='scale', kernel='poly', C=0.8, degree=3)
print("训练模型")
clf.fit(wineFeature, wineResult)
print("测试结果")
y_pred = clf.predict(testFeature)
print(y_pred)
print(testResult)
predWine = np.equal(y_pred, testResult)
correctCount = float(sum(map(lambda x: 1 if x else 0, predWine)))
print("正确样本数:%d,总测试样本数:%d,正确率:%g" % (correctCount, len(testResult), correctCount / len(testResult)))
