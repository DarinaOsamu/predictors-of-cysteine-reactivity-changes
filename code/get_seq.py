# -*- coding: utf-8 -*-


import pandas as pd

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
    

def main():
    
    #打开去冗余后的蛋白质序列fasta文件
    f=open('../source_data/CD_Hit.fasta')
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
    #print(seq)
    
    
    #加载位点数据
    Neg_site = read_data('../source_data/CD_Neg_Site.csv')
    Incre_site = read_data('../source_data/CD_Incre_Site.csv')
    Decre_site = read_data('../source_data/CD_Decre_Site.csv')
    
    #设置肽段格式，引入空缺符X
    keyChar='C'
    Len=10
    ModSeq=keyChar
    for i in range(Len):
        ModSeq='X'+ModSeq+'X'
    #print(ModSeq)

    #获取反应性上升的肽段
    Incre_seq=get_seq(Incre_site,seq,ModSeq,Len)
    Incre_site['sequence']=Incre_seq
    Incre_site['type']='Increased'
    #print(Incre_site.iloc[0:3,:])
    #print(Incre_site.shape)
    Incre_site.to_excel(excel_writer=r'../result/dataset/CD_Incre_Seq.xlsx',index=False)

    #获取反应性下降的肽段
    Decre_seq=get_seq(Decre_site,seq,ModSeq,Len)
    Decre_site['sequence']=Decre_seq
    Decre_site['type']='Decreased'
    #print(Decre_site.iloc[0:3,:])
    #print(Decre_site.shape)
    Decre_site.to_excel(excel_writer=r'../result/dataset/CD_Decre_Seq.xlsx',index=False)
    
    #获取反应性不变的肽段
    Unchanged_seq=get_seq(Neg_site,seq,ModSeq,Len)
    Neg_site['sequence']=Unchanged_seq
    Neg_site['type']='Unchanged'
    #print(Neg_site.iloc[0:3,:])
    #print(Neg_site.shape)
    Neg_site.to_excel(excel_writer=r'../result/dataset/CD_Unchanged_Seq.xlsx',index=False)
    
    #合并得到全体数据集
    Pos_seq=pd.concat([Incre_site,Decre_site],axis=0,ignore_index=True)
    #print(Pos_seq.iloc[0:3,:])
    #print(Pos_seq.shape)
    All_seq=pd.concat([Pos_seq,Neg_site],axis=0,ignore_index=True)
    #print(All_seq.iloc[0:3,:])
    #print(All_seq.shape)
    All_seq.to_excel(excel_writer=r'../result/dataset/CD_All_Seq.xlsx',index=False)
    

if __name__ == '__main__':

    main()