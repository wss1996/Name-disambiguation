# -*- coding: utf-8 -*-
from os.path import join
import codecs
import math
from collections import defaultdict as dd
from utils.embedding import EmbeddingModel
from utils.embedding import MySenteces1
from datetime import datetime
from utils.cache import LMDBClient
from utils import data_utils
from utils import settings
import re
import pandas as pd
import logging
from gensim.models import Word2Vec
import numpy as np
import random
from utils.data_utils import Singleton


start_time = datetime.now()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
EMB_DIM = 100



# class MySenteces():
#     def __init__(self, dirname):
#         self.dirname = dirname

#     def __iter__(self):
#     for fname in os.listdir(self.dirname):
#         for line in open(os.path.join(self.dirname, fname)):
#             yield line.split()
            
class MySenteces1():
    def __init__(self, dirname):
        self.LMDB_NAME = dirname
        self.lc = LMDBClient(self.LMDB_NAME)
        

    def __iter__(self):
        with self.lc.db.begin() as txn:
            for k in txn.cursor():
                author_feature = data_utils.deserialize_embedding(k[1])
                random.shuffle(author_feature)
                yield author_feature

class Word2vecModel:

    def __init__(self, name="sci_alldata"):
        self.model = None
        self.name = name


    def train(self, wf_name, size=EMB_DIM):
        data = MySenteces1('sci_all_data_feature')
        self.model = Word2Vec(data,size=100,negative =5, min_count=2, window=5,workers=16)
        self.model.save(join(settings.EMB_DATA_DIR, '{}.emb'.format(wf_name)))


    def load(self, name):
        self.model = Word2Vec.load(join(settings.EMB_DATA_DIR, '{}.emb'.format(name)))
        return self.model

    def project_embedding(self, tokens, idf=None):
        """
        weighted average of token embeddings
        :param tokens: input words
        :param idf: IDF dictionary
        :return: obtained weighted-average embedding
        """
        if self.model is None:
            self.load(self.name)
            print('{} embedding model loaded'.format(self.name))
        vectors = []
        sum_weight = 0
        for token in tokens:
            if not token in self.model.wv:
                continue
            weight = 1
            if idf and token in idf:
                weight = idf[token]
            v = self.model.wv[token] * weight
            vectors.append(v)
            sum_weight += weight
        if len(vectors) == 0:
            print('all tokens not in w2v models')
            # return np.zeros(self.model.vector_size)
            return None
        emb = np.sum(vectors, axis=0)
        emb /= sum_weight
        return emb




##将数据帧转化为列表字典，进一步变为二级字典
def data_todict(dataset):
    dataset_dict = dataset.to_dict(orient = 'records')
    data_dict = {}
    for pap in dataset_dict:
        pap['author'] = pap['author'].split('|')
        data_dict[pap['uid']] = pap 
    return data_dict

def dump_file_todict():
    ##采用分块读取的方法，主要用到参数chunksize，iterator参数（常用参数）
    yy = pd.read_csv('/home/wss/sites/disamb/sci_process/data/t_018_sci_disamb_string_precess.csv',
                       usecols = ['uid','author','title','abstract','keyword','org_name','pubyear','source'],sep = ',',iterator=True,encoding ='utf-8')

    # df = yy.get_chunk(1)
    # print(len(df))
    # print(df.columns)
    # print(df.head)
    loop = True
    chunkSize = 5000
    cnt = 0
    lc = LMDBClient('sci_all_data')
    while loop:
        try:
            chunk = yy.get_chunk(chunkSize)
            cnt += 1
            print('sci1800万论文存储了：%0.2f 万行'%(cnt*0.5))
            dataset_dict = chunk.to_dict(orient = 'records')
            for pap in dataset_dict:
                pap['author'] = pap['author'].split('|')
                pid_order = pap['uid']
                lc.set(pid_order, pap)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    lc.db.close()
    print("分块处理存取进lmdb数据库用时为：%0.2f s"%(time.time()-start))

def extract_author_features(pub):
    stopword = ['at','based','in','of','for','on','and','to','an','with','the','by','this','we','be','is','are','can','or','no','from','like']
    org = ''
    if type(pub["org_name"]) is str:
        org = pub['org_name']
    if type(pub["title"]) is str:
        title = pub['title']
    else:
        title = ''
    if type(pub["abstract"]) is str:
        abstract = pub["abstract"]
    else:
        abstract = ''
    if type(pub["source"]) is str:
        source = pub['source']
    else:
        source = ''
    if 'pubyear' in pub and pub['pubyear'] is not None:
        year = str(pub['pubyear'])
    else:
        year = ''
    pstr = org+ ' '+title+' '+source+' '+abstract+' '+year
    feature = re.sub('[^a-z0-9A-Z]', ' ',pstr).strip().lower().split() 
    feature = [word for word in feature if word not in stopword and len(word)>2]   
    return feature

def dump_features_to_cache():
    '''
    generate author features by raw publication data and dump to cache
    
    '''
    lc = LMDBClient('sci_all_data')
    lm = LMDBClient('sci_all_data_feature')
    cnt = 0
    with lc.db.begin() as txn:
        for k in txn.cursor():
            cnt += 1
            pid = k[0].decode()
            paper = data_utils.deserialize_embedding(k[1])
            if len(paper["author"]) > 100:
                print(cnt, pid, len(paper["author"]))
                continue
            features = extract_author_features(paper)
            if cnt % 10000 == 0:
                print('已经提取：%d 万篇论文'%(cnt/10000))
            lm.set(pid,features)
    lm.db.close()
    lc.db.close()
            
            

def cal_feature_idf():
    """
    calculate word IDF (Inverse document frequency) using publication data
    """
    feature_dir = join(settings.DATA_DIR, 'global')
    counter = dd(int)
    cnt = 0
    LMDB_NAME = 'sci_all_data_feature'
    lc = LMDBClient(LMDB_NAME)
    author_cnt = 0
    with lc.db.begin() as txn:
        for k in txn.cursor():
#            print(k[0])
            features = data_utils.deserialize_embedding(k[1])
#            print(features)
            if author_cnt % 10000 == 0:
                print(author_cnt, features[0], counter.get(features[0]))
            author_cnt += 1
            for f in features:
                cnt += 1
                counter[f] += 1
    idf = {}
    for k in counter:
        idf[k] = math.log(cnt / counter[k])
    data_utils.dump_data(dict(idf), feature_dir, "feature_idf.pkl")

def dump_paper_embs():
    """
    dump author embedding to lmdb
    author embedding is calculated by weighted-average of word vectors with IDF
    """
    emb_model = EmbeddingModel.Instance()
    idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf.pkl')
    print('idf loaded')
    LMDB_NAME_FEATURE = 'sci_all_data_feature'
    lc_feature = LMDBClient(LMDB_NAME_FEATURE)
    LMDB_NAME_EMB = "sci_allpaper.emb.weighted"
    lc_emb = LMDBClient(LMDB_NAME_EMB)
    model = emb_model.load(emb_model.name)
    cnt = 0
    with lc_feature.db.begin() as txn:
        for k in txn.cursor():
            if cnt % 10000 == 0:
                print('cnt', cnt, datetime.now()-start_time)
            cnt += 1
            pid_order = k[0].decode('utf-8')
            features = data_utils.deserialize_embedding(k[1])
            cur_emb = emb_model.project_embedding(features, idf,model)
            if cur_emb is not None:
                lc_emb.set(pid_order, cur_emb)

                



if __name__ == '__main__':
    """
    some pre-processing
    """
    dump_file_todict()##将1800万sci论文写入lmdb
    dump_features_to_cache()##将每篇论文特征提取之后写入lmdb
    cal_feature_idf() ##计算每个词的idf权重系数

    emb_model = Word2vecModel()# 训练词向量
    emb_model.train('sci_alldata')  
    
    dump_paper_embs()##将idf加权之后的论文嵌入写入lmdb
    print('done', datetime.now()-start_time)






