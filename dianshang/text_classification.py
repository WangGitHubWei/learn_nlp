#!/usr/bin/python
#coding:utf-8

"""
@author: ww
@software: PyCharm
@file: text_classification.py
@time: 2019/8/14 14:39
"""

'''
电商用户评论文本分类
'''
import jieba
import gensim
import scipy
import numpy
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB

#加载词典
stop = ''
with open('stopwords.txt','r',encoding='gbk',errors='ignore') as s:
    for line in s:
        line = line.strip()
        stop += line


#读取文本变量数据
dataList = [] #特征
tagList = [] #标签
Count = 0
fobjRead =open('1578698_content.txt','r',encoding='utf-8')

for row in fobjRead:
    if Count >= 5000:
        break
    score = int(row[2])
    if score >= 4:
        flag = 1
    elif score >= 3:
        flag = 2
    else:
        flag =3
    if flag in [1,3]:
        content = row.strip("\n").split(':#:')[1].replace(' ','')
        #分词　停用词　去重等预处理
        wordList = jieba.cut(content, cut_all=False)
        termsAll = list(set([term for term in wordList if term not in stop]))
        dataList.append(termsAll)
        tagList.append(str(flag))
        Count = Count + 1
fobjRead.close()

#文本特征向量
wordDict = gensim.corpora.Dictionary(dataList)
corpus = [wordDict.doc2bow(doc) for doc in dataList]

#文本特征向量转为sklearn 可以识别的稀疏矩阵
data = []
rows = []
cols = []
line_count = 0
for line in corpus:
    for elem in line:
        rows.append(line_count)
        cols.append(elem[0])
        data.append(elem[1])
    line_count = line_count + 1
matrix = scipy.sparse.csr_matrix((data,(rows,cols))).toarray()
rarray = numpy.random.random(size=line_count)
#print(matrix)
#print(rarray)
'''
[[1 1 1 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]
[0.26456867 0.31671304 0.24109927 ... 0.68329533 0.81079462 0.91965377]
'''

# 划分训练集　测试集、
train_set = []
train_tag = []
test_set = []
test_tag = []
totalCount = sum([500,500])
#print(totalCount)
posCount, negCount = [500,500]
posNow, negNow =0, 0
recordCount = 0
for i in range(line_count):
    if rarray[i] < 0.8 and (posNow + negNow) < totalCount:
        if tagList[i] == "1" and posNow < posCount:
            train_set.append(matrix[i,:])
            train_tag.append(tagList[i])
            posNow = posNow + 1
        elif tagList[i] == "3" and negNow < posCount:
            train_set.append(matrix[i,:])
            train_tag.append(tagList[i])
            negNow = negNow + 1
        else:
            test_set.append(matrix[i,:])
            test_tag.append(tagList[i])
    else:
        test_set.append(matrix[i,:])
        test_tag.append(tagList[i])
del matrix
del rarray

print(train_set)
print(train_tag)
print('------------------------------- \
      -------------')
print(test_set)
print(test_tag)

#建模

#决策树
clf = DecisionTreeClassifier()
clf.fit(train_set, train_tag)
clf_predict_test = clf.predict(test_set)
print(sklearn.metrics.classification_report(test_tag,clf_predict_test))
'''
              precision    recall  f1-score   support

           1       0.74      0.52      0.61      2918
           3       0.28      0.50      0.36      1082

    accuracy                           0.52      4000
   macro avg       0.51      0.51      0.48      4000
weighted avg       0.61      0.52      0.54      4000
'''

#朴素贝叶斯
clf1 = BernoulliNB()
clf1.fit(train_set, train_tag)
clf1_predict_test = clf1.predict(test_set)
print(sklearn.metrics.classification_report(test_tag,clf1_predict_test))
'''
             precision    recall  f1-score   support

           1       0.74      0.36      0.49      2918
           3       0.28      0.65      0.39      1082

    accuracy                           0.44      4000
   macro avg       0.51      0.51      0.44      4000
weighted avg       0.61      0.44      0.46      4000

'''