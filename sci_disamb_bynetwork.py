import re
from gensim.models import word2vec
import gensim
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import codecs
import json
from os.path import join
import pickle
import os
import nltk
from utils.cache import LMDBClient
# from sklearn.cluster import AgglomerativeClustering
import multiprocessing
import pandas as pd
import csv
import time
from datetime import datetime
import random

start_time = datetime.now()

################# Load and Save Data ################

def load_json(rfdir, rfname):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        return json.load(rf)


def dump_json(obj, wfpath, wfname, indent=None):
    with codecs.open(join(wfpath, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)


def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)


def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)

    

##抽取训练集的结果作评分测试

def value_dict(dataset):
    train_data_n ={}
    cnt =0
    for i,(k,v) in enumerate(dataset.items()):
        if len(v) >= 50 and len(v) <=51:
            train_data_n[k] = v
            cnt += 1
        if cnt == 1000:
            break
    return train_data_n

################# Save Paper Features ################


def load_stopwords():
    """
    加载停用词
    :return:
    """
    sw = set()
    with open("train/stop_word.txt", 'r') as f:
        for line in f.readlines():
            sw.add(line[:-1])
    return sw


# 词形还原、转为小写
def word_lemmatize(context):
    stopword = ['at','based','in','of','for','on','and','to','an','using','with','the','by','we','be','is','are','can','or','no','from','like']
    stopword1 = load_stopwords() 
    context = re.sub('[^a-zA-Z]', ' ', context).strip().lower().split()
    for word in context:
        if word not in stopword and word not in stopword1 and len(word)> 2:
            yield word

def save_relation(name_pubs_raw, name,number): # 保存论文的各种feature
    
#     name_pubs_raw = load_json('/home/wss/data/disk1/sci_disamb_name_50', name_pubs_raw)
    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
    f1 = open ('gene/paper_author_{}.txt'.format(number),'w',encoding = 'utf-8')
    f2 = open ('gene/paper_conf_{}.txt'.format(number),'w',encoding = 'utf-8')
    f3 = open ('gene/paper_word_{}.txt'.format(number),'w',encoding = 'utf-8')
    f4 = open ('gene/paper_org_{}.txt'.format(number),'w',encoding = 'utf-8')

    taken = name.split()
    name = ''.join(taken)
    name_reverse = ''.join(taken[::-1])
    
    authorname_dict={}
    ptext_emb = {}  
    tcp=0  
    for i,pid in enumerate(name_pubs_raw):
        
        pub = name_pubs_raw[pid]
        
        #save authors 保留作者特征，每篇文章对应的作者名
        for author in pub["author"]:
            authorname = re.sub(r,'', author).lower().strip()
            taken = authorname.split(" ")
            
            if len(taken)==2: ##检测目前作者名是否在作者词典中
                authorname = ''.join(taken)
                authorname_reverse = ''.join(taken[::-1]) 
            
                if authorname not in authorname_dict:
                    if authorname_reverse not in authorname_dict:
                        authorname_dict[authorname]=1
                    else:
                        authorname = authorname_reverse 
            else:
                authorname = authorname.replace(" ","")
            
            if authorname!=name and authorname!=name_reverse:
                f1.write(pid + '\t' + authorname + '\n')
                
                
         #save org 待消歧作者的机构名
        org=""
        if type(pub['org_name']) is str: 
            org = pub['org_name']
        else: 
            org = ''
        pstr = ' '.join(word_lemmatize(org)).strip().lower().split()
        pstr=set(pstr)
        for word in pstr:
            f4.write(pid + '\t' + word + '\n')
        
        #save SOURCETITLE
        source=""
        if type(pub['source']) is not float: 
            source = pub['source']
        else: 
            source = ''
        pstr = ' '.join(word_lemmatize(source.strip())).strip().lower().split()
        for word in pstr:
            f2.write(pid + '\t' + word + '\n')
        if len(pstr)==0:
            f2.write(pid + '\t' + 'null' + '\n')
 

        #save text
        pstr = ""    
        keyword=""
        if pub["keyword"] is not None and type(pub["keyword"]) is not float:
            keyword = pub['keyword']
        else:
            keyword=""
        pstr = pstr + pub["title"]
        pstr = ' '.join(word_lemmatize(pstr)).strip().lower().split()
        for word in pstr:
            f3.write(pid + '\t' + word + '\n')
        
#         LMDB_NAME_EMB = "sci_allpaper.emb.weighted"
#         lc_emb = LMDBClient(LMDB_NAME_EMB)
        
#             #print ('outlier:',pid,pstr)
#         if lc_emb.get(pid) is not None:
#             ptext_emb[pid] = lc_emb.get(pid)
#         else:
#             ptext_emb[pid] = np.zeros(100)
#             tcp += 1 

#     #  ptext_emb: key is paper id, and the value is the paper's text embedding
#     dump_data(ptext_emb,'gene','ptext_emb_{}.pkl'.format(number))
    
#     # the paper index that lack text information
#     dump_data(tcp,'gene','tcp_{}.pkl'.format(number))
            
 
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    
    

class MetaPathGenerator:
    def __init__(self):
        self.paper_author = dict()
        self.author_paper = dict()
        self.paper_org = dict()
        self.org_paper = dict()
        self.paper_conf = dict()
        self.conf_paper = dict()

    def read_data(self, dirpath, number):
        temp=set()

        with open(dirpath + "/paper_org_{}.txt".format(number), encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)                       
        for line in temp: 
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p, a = toks[0], toks[1]
                    if p not in self.paper_org:
                        self.paper_org[p] = []
                    self.paper_org[p].append(a)
                    if a not in self.org_paper:
                        self.org_paper[a] = []
                    self.org_paper[a].append(p)
        temp.clear()

              
        with open(dirpath + "/paper_author_{}.txt".format(number), encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)                       
        for line in temp: 
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p, a = toks[0], toks[1]
                    if p not in self.paper_author:
                        self.paper_author[p] = []
                    self.paper_author[p].append(a)
                    if a not in self.author_paper:
                        self.author_paper[a] = []
                    self.author_paper[a].append(p)
        temp.clear()
        
                
        with open(dirpath + "/paper_conf_{}.txt".format(number), encoding='utf-8') as pcfile:
            for line in pcfile:
                temp.add(line)                       
        for line in temp: 
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p, a = toks[0], toks[1]
                    if p not in self.paper_conf:
                        self.paper_conf[p] = []
                    self.paper_conf[p].append(a)
                    if a not in self.conf_paper:
                        self.conf_paper[a] = []
                    self.conf_paper[a].append(p)
        temp.clear()
                    
    
    def generate_WMRW(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')
        for paper0 in self.paper_conf: 
            for j in range(0, numwalks): #wnum walks
                paper=paper0
                outline = ""
                i=0
                while(i<walklength):
                    i=i+1    
                    if paper in self.paper_author:
                        authors = self.paper_author[paper]
                        numa = len(authors)
                        authorid = random.randrange(numa)
                        author = authors[authorid]
                        
                        papers = self.author_paper[author]
                        nump = len(papers)
                        if nump >1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper           
                        
                    if paper in self.paper_org:
                        words = self.paper_org[paper]
                        numw = len(words)
                        wordid = random.randrange(numw) 
                        word = words[wordid]
                    
                        papers = self.org_paper[word]
                        nump = len(papers)
                        if nump >1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper  
                            
                outfile.write(outline + "\n")
                
        outfile.close()
        
################# Compare Lists ################
###拥有相同的东西/所有的东西 

def tanimoto(p,q):
    c = [v for v in p if v in q]
    return float(len(c) / (len(p) + len(q) - len(c)))



################# Paper similarity ################

def get_paper_dict(pathfile): ##将论文的各种特征转化为映射表字典
    dirpath = 'gene/'
    paper_org = {}
    
    temp=set()
    with open(dirpath + pathfile, encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)                       
    for line in temp: 
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_org:
                paper_org[p] = []
            paper_org[p].append(a)
    temp.clear()
    return paper_org

def generate_pair(pubs,outlier,number): ##求匹配相似度
#     dirpath = 'gene'

    paper_org = get_paper_dict('paper_org_{}.txt'.format(number))
    paper_conf = get_paper_dict('paper_conf_{}.txt'.format(number))
    paper_author = get_paper_dict('paper_author_{}.txt'.format(number))
    paper_word = get_paper_dict('paper_word_{}.txt'.format(number))

    paper_paper = np.zeros((len(pubs),len(pubs)))
    for i,pid in enumerate(pubs):
        if i not in outlier:
            continue
        for j,pjd in enumerate(pubs):
            if j==i:
                continue
            ca=0
            cv=0
            co=0
            ct=0
          
            if pid[0] in paper_author and pjd[0] in paper_author:
                ca = len(set(paper_author[pid[0]])&set(paper_author[pjd[0]]))*1.5
            if pid[0] in paper_conf and pjd[0] in paper_conf and 'null' not in paper_conf[pid[0]]:
                cv = tanimoto(set(paper_conf[pid[0]]),set(paper_conf[pjd[0]]))
#                cv = len(set(paper_conf[pid])&set(paper_conf[pjd]))/3
            if pid[0] in paper_org and pjd[0] in paper_org:
                co = tanimoto(set(paper_org[pid[0]]),set(paper_org[pjd[0]]))
#                co = len(set(paper_org[pid])&set(paper_org[pjd]))/3
            if pid[0] in paper_word and pjd[0] in paper_word:
                ct = len(set(paper_word[pid[0]])&set(paper_word[pjd[0]]))/3
                    
            paper_paper[i][j] =ca+cv+co+ct
            
    return paper_paper


def disambiguate(name_pubs,number):
    print('Run task (%s)...' % (os.getpid()))
    start1 = time.time()
    lc1 = LMDBClient('sci_all_data')
    result ={}
    for n,name in enumerate(name_pubs):
        pubs = name_pubs[name]##存储某一人名下的所有文章
        print(n,name,len(pubs))  

        if len(pubs)==0:
            result[name] = []
            continue
        result1=[]
        if len(pubs)<=5:
            result[name] = []
            for i,pid in enumerate(pubs):
                result1.append(pid[0])
            result[name].append(result1)
            continue
        
        ##保存关系
        ###############################################################
        name_pubs_raw = {}
        for i,pid in enumerate(pubs):
            paper = lc1.get(pid[0])
            paper['org_name'] = pid[1]
            paper.pop('abstract')
            paper.pop('uid')
            name_pubs_raw[pid[0]] = paper
        save_relation(name_pubs_raw, name,number)  
#         print('save features down')
        ###############################################################

        ##元路径游走类
        mpg = MetaPathGenerator()
        mpg.read_data("gene",number)
#         print('path down')
        ###############################################################

        ##论文关系表征向量(关系特征嵌入),采用了bagging思想
        all_embs=[]
        rw_num =3
        cp=set()##孤立节点
        for k in range(rw_num):
            mpg.generate_WMRW("gene/RW_{}.txt".format(number),5,10) #生成路径集
            sentences = word2vec.Text8Corpus(r'gene/RW_{}.txt'.format(number))
            model = word2vec.Word2Vec(sentences, size=100,negative =25, min_count=1, window=10)
            embs=[]
            for i,pid in enumerate(pubs):
                if pid[0] in model.wv:
                    embs.append(model.wv[pid[0]])
                else:
                    cp.add(i)
                    embs.append(np.zeros(100))
            all_embs.append(embs)
        all_embs= np.array(all_embs)
#         print('real emb down')
        ###############################################################

        ##论文文本表征向量
        ###############################################################   
#         ptext_emb=load_data('gene','ptext_emb_{}.pkl'.format(number))
#         tcp=load_data('gene','tcp_{}.pkl'.format(number))
#         tembs=[]
#         for i,pid in enumerate(pubs):
#             tembs.append(ptext_emb[pid[0]])
# #         print('paper emb down')
        ############################################################### 


        ##网络嵌入向量相似度
        sk_sim = np.zeros((len(pubs),len(pubs)),dtype='float16')
        for k in range(rw_num):
            sk_sim = sk_sim + pairwise_distances(all_embs[k],metric="cosine")
        sk_sim =sk_sim/rw_num    

        ##文本相似度
#         t_sim = pairwise_distances(tembs,metric="cosine")
#         if tcp >= len(pubs)/2: 
        sim = np.array(sk_sim)
#         else:
#             w=1#相似度矩阵融合权重 
#             sim = (np.array(sk_sim) + w*np.array(t_sim))/(1+w)

        ##实现消歧聚类
        ###############################################################
        pre = DBSCAN(eps = 0.2, min_samples = 1 ,metric ="precomputed").fit_predict(sim)

       ##离散点
        outlier=set()
        for i in range(len(pre)):
            if pre[i]==-1:
                outlier.add(i)
        for i in cp:
            outlier.add(i)


        ## (给每一个离群节点打上标签,基于tanimoto相似度矩阵)
        paper_pair = generate_pair(pubs,outlier,number)
        paper_pair1 = paper_pair.copy()
        K = len(set(pre))
        for i in range(len(pre)):
            if i not in outlier:
                continue
            j = np.argmax(paper_pair[i])
            while j in outlier:
                paper_pair[i][j]=-1
                j = np.argmax(paper_pair[i])
            if paper_pair[i][j]>=1.5:
                pre[i]=pre[j]
            else:
                pre[i]=K
                K=K+1

        ## find nodes in outlier is the same label or not
        ## 将各个离群节点通过相似度匹配来打上相同标签，相似阈值为1.5
        for ii,i in enumerate(outlier):
            for jj,j in enumerate(outlier):
                if jj<=ii:
                    continue
                else:
                    if paper_pair1[i][j]>=1.5:
                        pre[j]=pre[i]

        ##存储消歧预测结果
        result[name]=[]
        for lab in set(pre):
            sameauthor = []
            for index,lab1 in enumerate(pre):
                if lab == lab1:
                    sameauthor.append(pubs[index][0])
            result[name].append(sameauthor)
        print('消歧之后的作者数量:%d'%(len(result[name])))
    dump_json(result, "output", "sci_result_1_10_all_{}.json".format(number),indent =4)
    print('task %s run %0.2f seconds.' % (os.getpid(), (time.time() - start1)))
    
    
def chunks(data, SIZE=1000):
    for i in range(0, len(data), SIZE):
        yield dict(list(data.items())[i:i+SIZE])


##加载需要消歧的数据
name_pubs = load_json('data', 'sci_author_data_1_10_all.json')
print('同名论文集数量为：',len(name_pubs))


if __name__ =='__main__':
    processor = 16
    p = multiprocessing.Pool(processor)
    for i,item in enumerate(chunks(name_pubs,720000)):
#         name_pub =item
#         if i in [0,1,2,3,4]:
        p.apply_async(disambiguate, args=(item,i,))
    p.close()
    p.join()
    print('论文数量大于1小于10的作者名消歧程序 总用时 ：',datetime.now()-start_time)