# -*- coding: utf-8 -*-
import requests

#获取蛋白质编号
#打开去冗余后的蛋白质序列fasta文件
f=open('../source_data/CD_Hit.fasta')
#建立列表存放蛋白质编号
accession_list=[]
for line in f:
    if line.startswith('>'):
        #以蛋白质编号为键
        name=line.replace('>', '').split('|')[1]
        accession_list+=[name]
f.close()
print(accession_list)

with open("../source_data/IUPRED2A.txt", "w") as f:
    for accession in accession_list:
        title='>xx|'+accession
        url="http://iupred2a.elte.hu/iupred2a/"+accession
        response = requests.get(url)
        #print(response.status_code)
        #print(response.text)

        f.write(title)
        f.write('\n')
        f.write(response.text)
        f.write('\n')
        break
