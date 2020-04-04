# -*- coding: utf-8 -*-
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import jieba
data = pd.read_csv(r"train_data.txt", encoding = 'utf-8', sep = '-', names=['A','B','C'])
f=lambda x:' '.join(jieba.cut(x))
data['分词']=data['C'].apply(f)
content = data['分词'].values
#返回的content就是那些语句的列表
label = data['B'].values
#print(label)
#返回标签的列表
from sklearn.model_selection import train_test_split
train_content,test_content,train_label,test_label = train_test_split(content, label, test_size = 0.01)
#print(train_content)，也是列表，内容就是content的那些东西
vectorizer = CountVectorizer()
content_train_termcounts = vectorizer.fit_transform(train_content)
#获取文本特征,就是词频特征，或者说是词频矩阵content_train_termcounts
#vectorizer这时候被实例化为特征向量（存储了特征值）
#print(vectorizer.get_feature_names())
tfidf_transformer = TfidfTransformer()
content_train_tfidf = tfidf_transformer.fit_transform(content_train_termcounts)
#将词频矩阵转化为ti-idf矩阵
classifier = MultinomialNB().fit(content_train_tfidf, train_label)
#进行训练
content_input_termcounts  = vectorizer.transform(test_content)
content_input_tfidf = tfidf_transformer.transform(content_input_termcounts)
#输入测试值，进行测试
predicted_categories = classifier.predict(content_input_tfidf)
print('predicted : ', predicted_categories)
print('accuracy : ',accuracy_score(test_label, predicted_categories))
