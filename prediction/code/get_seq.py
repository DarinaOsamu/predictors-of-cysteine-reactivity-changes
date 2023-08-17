# -*- coding: utf-8 -*-


import pandas as pd
import re

def read_data(file_path):
    
    dataset = pd.read_csv(file_path)
    '''
    #查看并显示前三条记录
    print(dataset.iloc[0:3,:])
    print('-' * 30)
    
    #查看样本数和特征数
    print(dataset.shape)
    print('-' * 30)
    '''
    return(dataset)

def get_seq(site,seq,ModSeq,Len):
    #print(site.shape[0])
    site_seq=[]
    for i in range(site.shape[0]):
        outputSeq=list(ModSeq)
        tmpSeq=list(seq[site.iloc[i,0]])
        position=site.iloc[i,1]
        #上游肽段
        for j in range(Len):
            if position-1==j:
                break
            outputSeq[Len-1-j]=tmpSeq[position-2-j]
        #下游肽段
        for j in range(Len):
            if position+j==len(tmpSeq):
                break
            outputSeq[Len+1+j]=tmpSeq[position+j]
        #把字符列表转化为字符串
        outputSeq=''.join(outputSeq)
        site_seq.append(outputSeq)
    #print(site_seq)
    return(site_seq)
    

###主函数main()----------

#打开蛋白质序列fasta文件
f=open('../source_data/uniprot-download.fasta')
#建立字典
seq={}
for line in f:
    if line.startswith('>'):
        #以蛋白质编号为键
        name=line.replace('>', '').split('|')[1]
        seq[name]=''
    else:
        seq[name]+=line.replace('\n', '').strip()
f.close()
print(seq.__len__())


#找到C位点位置
#建立位点Dataframe
C_site=pd.DataFrame(columns=['accession','position'])
number=0
for key in seq:
    accession=key
    value=seq[key]
    for i in range(len(value)):
        if value[i]=='C':
            position=i+1
            number += 1
            C_site.loc[len(C_site)]=[accession,position]
print(number)
#print(C_site)

#设置肽段格式，引入空缺符X
keyChar='C'
Len=10
ModSeq=keyChar
for i in range(Len):
    ModSeq='X'+ModSeq+'X'
#print(ModSeq)

#获取肽段
All_seq=get_seq(C_site,seq,ModSeq,Len)
C_site['sequence'] = All_seq
print(C_site.iloc[0:3, :])
print(C_site.shape)
# 把位点数据写入csv
C_site.to_csv('../source_data/CD_All_seq.csv', index=False)

#删除含有20种氨基酸之外的氨基酸残基的肽段
#seq只能含有ACDEFGHIKLMNPQRSTVWYX
p='^[ACDEFGHIKLMNPQRSTVWYX]+$'
drop_index=[]
n=0
for i in range(C_site.shape[0]):
    sequence=C_site.loc[i,'sequence']
    if bool(re.match(p,sequence))==False:
        drop_index+=[i]
        n+=1
print(drop_index)
print(n)
print(C_site.iloc[drop_index,:])
C_Drop_seq=C_site.drop(index=drop_index)
print(C_Drop_seq.iloc[0:3, :])
print(C_Drop_seq.shape)
# 把位点数据写入csv
C_Drop_seq.to_csv('../source_data/Drop_seq.csv', index=False)
