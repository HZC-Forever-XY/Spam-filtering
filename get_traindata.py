# -*- coding: utf-8 -*-
fp = open('traindata.txt', mode='r', encoding='UTF-8')
data=''
#s=fp.read()
list=[]
for line in fp.readlines():
    list.append(line)
#print(list)
result_list=[]
i=1
for a in list:
    STR=str(i)+'-'+'     '+a[0]+'-'+'     '+a[2:]
    result_list.append(STR)
    #print(str)
    i=i+1
S=''.join(result_list)
with open('traindata.txt',"w",encoding="utf-8") as f:
    f.write(S)
f.close()
fp.close()

