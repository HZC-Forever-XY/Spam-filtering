# -*- coding: utf-8 -*-
import pandas as pd
import jieba
import jieba.posseg as pseg
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
data = pd.read_csv(r"testdata.txt", encoding = 'utf-8', sep = '-', names=['A','B','C'])
g=lambda doc: ''.join(w.word for w in pseg.cut(doc) if w.flag != 'x')
data['D'] = data['C'].apply(g)
#去掉没有意义的词汇
f=lambda x:jieba.cut(x,cut_all=False)
data['分词']=data['D'].apply(f)
test_content = data['分词'].values
test_label = data['B'].values
Content = [list(i) for i in test_content ]
test_content = Content
classfiy_model = joblib.load('finalized_model.m')
#从本地加载模型
predicted_categories = classfiy_model.classfiy_message(test_content)
#将需要鉴定的短信输入到模型
category = {
    0:'正常短信',
    1:'垃圾短信'
}
test_message = data['C'].values
#print('predic',predicted_categories)
print('accuracy : ',accuracy_score(test_label,predicted_categories))
#输出准确度
for sentence, pre, real in zip(test_message[:5], predicted_categories[:5],test_label[:5]):
    print('\n短信内容: ', sentence, '\nPredicted 分类: ', category[pre], "真实值: ", category[real])
#可视化，输出前5条短信的内容与结果

