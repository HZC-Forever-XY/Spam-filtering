# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import jieba
import jieba.posseg as pseg
import my_functions

data = pd.read_csv(r"traindata.txt", encoding = 'utf-8', sep = '-', names=['A','B','C'])
g=lambda doc: ''.join(w.word for w in pseg.cut(doc) if w.flag != 'x')
data['D'] = data['C'].apply(g)
#去掉没有意义的词汇
f=lambda x:jieba.cut(x,cut_all=False)
data['分词']=data['D'].apply(f)
#进行分词
content = data['分词'].values
label = data['B'].values
Content = [list(i) for i in content ]
content = Content
#将原来的类型转变为可视化的列表类型
content = my_functions.Filter_vocabulary(content)
#滤词
train_content,test_content,train_label,test_label = train_test_split(content, label, test_size = 0.01)
print('get train content!!!')
#将原始数据分为用来训练的部分和用来测试的部分
wordlist = my_functions.GetWordList(train_content)
print('get the word list!!!')
#获得词汇列表，储存了所有可能出现的单词
wordmatrix = my_functions.word_matrix(train_content,wordlist)
print('get the word matrix!!!')
#获得单词矩阵，每一行代表的是某条短信的单词在词汇表中是否出现
#行数就是训练的短信的条数
print('begin to train!!!')
classfiy = my_functions.classfication_model(wordmatrix,wordlist,train_label)
#进行训练
'''print('wordlist : ',wordlist)
print('wordmatix : ',wordmatrix)
print('trainlabel : ',train_label)
print('Posterior_probability_matrix : ',classfiy.Posterior_probability_matrix)
print('Priori_probability : ',classfiy.Priori_probability)
调试用的语句
'''
test_matrix = my_functions.word_matrix(test_content, wordlist)
#测试语句的单词矩阵
result=[]
for i in range(len(test_matrix[:,0])):
    result.append(classfiy.classfication(test_matrix[i,:]))
#每个测试语句代入模型进行测试
print('accuracy : ',accuracy_score(test_label, result))
#输出准确度
from sklearn.externals import joblib
filename = 'finalized_model.m'
joblib.dump(classfiy, filename)
#保存模型到本地

