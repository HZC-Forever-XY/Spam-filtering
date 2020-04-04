# -*- coding: utf-8 -*-
import numpy
import math
def GetWordList(data):
    #获得单词列表，里面包括了整个数据集所涉及的所有词汇
    Data = data
    wordlist = set([])
    for sentence in Data:
        wordlist = wordlist | set(sentence)
    return list(wordlist)

def word_matrix(content,wordlist):
    #获得单词矩阵
    wordmatrix=[]
    for setence in content:
        #遍历每一条短信
        setence_vec = [0]*len(wordlist)
        setence = list(setence)
        for word in setence:
            #遍历每条短信中的每一个单词
            if word in wordlist:
                setence_vec[wordlist.index(word)] = 1
                #如果这个单词在单词表里面，则对应位置的列表元素值改为1
                #列表初始值为0
        wordmatrix.append(setence_vec)
    return numpy.array(wordmatrix)

def setence_matrix(setence,wordlist):
    #获得语句矩阵
    setence_vec = [0]*len(wordlist)
    for word in setence:
        if word in wordlist:
            setence_vec[wordlist.index(word)] = 1
    return  setence_vec

class classfication_model:
    #关键类，获得训练模型
    def __init__(self,wordmatrix,wordlist,label):
        self.wordmatrix = wordmatrix
        self.wordlist = wordlist
        self.label = label
        self.Priori_probability = math.log(sum(label)/len(label))
        #是垃圾短信的先验概率，这里所有的概率都化为对数形式，将乘法运算化为加法运算
        def Posterior_probability(wordmatrix,label):
            #计算后验概率，例如：在为垃圾短信的条件下，出现xx特征的概率
            Posterior_probability_matrix = []
            label1 = [i for i,j in enumerate(label) if j==1]
            label0 = [i for i,j in enumerate(label) if j==0]
            wordmatrix_1 = wordmatrix[label1,:]
            #得到垃圾短信的单词矩阵
            wordmatrix_0 = wordmatrix[label0,:]
            #得到非垃圾短信的单词矩阵
            P_1_1 = [math.log((sum(wordmatrix_1[:,i])+1)/(len(wordmatrix_1[:,i])+2)) for i in range(len(wordmatrix_1[0,:]))]
            #列表，代表当label=1时候，特征也为1的概率
            #print('P11',P_1_1)
            P_0_1 = [math.log(1-(sum(wordmatrix_1[:,i])+1)/(len(wordmatrix_1[:,i])+2)) for i in range(len(wordmatrix_1[0,:]))]
            #print('P01',P_0_1)
            P_1_0 = [math.log((sum(wordmatrix_0[:,i])+1)/(len(wordmatrix_0[:,i])+2)) for i in range(len(wordmatrix_0[0,:]))]
            #print('P10', P_1_0)
            P_0_0 = [math.log(1-(sum(wordmatrix_0[:, i])+1) / (len(wordmatrix_0[:, i])+2)) for i in range(len(wordmatrix_0[0, :]))]
            #分子加1是做了拉普拉斯平滑
            Posterior_probability_matrix.append(P_1_1)
            Posterior_probability_matrix.append(P_0_1)
            Posterior_probability_matrix.append(P_1_0)
            Posterior_probability_matrix.append(P_0_0)
            #为了后续操作方便，全部放入Posterior_probability_matrix矩阵
            return numpy.array(Posterior_probability_matrix)
        self.Posterior_probability_matrix = Posterior_probability(wordmatrix,label)

    def classfication(self,sentence_matrix):
        #判断是否为垃圾短信
        P1 = self.Priori_probability
        #垃圾短信的先验概率
        P0 = 1- self.Priori_probability
        #非垃圾短信的先验概率
        label_1 = [i for i, j in enumerate(sentence_matrix) if j == 1]
        #特征为1的序号
        label_0 = [i for i, j in enumerate(sentence_matrix) if j == 0]
        #特征为0的序号
        for i in label_1:
            P1 =P1+self.Posterior_probability_matrix[0,i]
        for i in label_0:
            P1 =P1+self.Posterior_probability_matrix[1,i]
        #为垃圾短信的概率，由于之前取过对数，因此直接化乘法为加法
        for i in label_1:
            P0 = P0 + self.Posterior_probability_matrix[2, i]
        for i in label_0:
            P0 = P0+self.Posterior_probability_matrix[3, i]
        if P1>P0:
            #如果是垃圾短信的概率大，返回1
            return 1
        else:
            #否则返回0
            return 0

    def classfiy_message(self,test_content):
        #输入文本列表，直接返回预测列表
        test_matrix = word_matrix(test_content,self.wordlist)
        predicted_categorys =  []
        for i in range(len(test_matrix[:,0])):
            predicted_categorys.append(self.classfication(test_matrix[i,:]))
        return predicted_categorys

def Filter_vocabulary(content):
    #除掉训练内容中那些没有实际意义的词汇，例如“之所以”之类
    with open('stopwords.txt', 'r', encoding='UTF-8') as rf:
        str = rf.read()
    new_content = []
    for sentence in content:
        new_sentence = []
        for word in sentence:
            if word not in str.splitlines():
                new_sentence.append(word)
        new_content.append(new_sentence)
    rf.close()
    return new_content








