# -*- coding: utf-8 -*-

# import logging
from os.path import join
from gensim.models import Word2Vec
import numpy as np
import random
from utils.cache import LMDBClient
from utils import data_utils
from utils.data_utils import Singleton
from utils import settings
import gensim

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
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
        self.lc = LMDBClient(LMDB_NAME)
        

    def __iter__(self):
        with self.lc.db.begin() as txn:
            for k in txn.cursor():
                author_feature = data_utils.deserialize_embedding(k[1])
                random.shuffle(author_feature)
                yield author_feature

                
@Singleton
class EmbeddingModel:

    def __init__(self, name="sci_alldata"):
#         self.model = gensim.models.KeyedVectors.load_word2vec_format('word2vec_tsw/Aword2vec_sci_all.bin.gz', binary=True)
        self.name = name
#         self.model = Word2Vec.load('/home/wss/sites/disamb/OAG_first/word2vec/Aword2vec.model')
        self.model = None



    def train(self, wf_name, size=EMB_DIM):

        data = MySentences1('sci_all_data_feature')
        self.model = Word2Vec(data,size=100,negative =5, min_count=2, window=5,workers=5)
        self.model.save(join(settings.EMB_DATA_DIR, '{}.emb'.format(wf_name)))


    def load(self, name):
        model = Word2Vec.load(join(settings.EMB_DATA_DIR, '{}.emb'.format(name)))
        return model

    def project_embedding(self, tokens, idf=None,model = None):
        """
        weighted average of token embeddings
        :param tokens: input words
        :param idf: IDF dictionary
        :param model： word2vec
        :return: obtained weighted-average embedding
        """
        if model is None:
            self.load(self.name)
            print('{} embedding model loaded'.format(self.name))
        vectors = []
        sum_weight = 0
        for token in tokens:
            if not token in model.wv:
                continue
            weight = 1
            if idf and token in idf:
                weight = idf[token]
            v = model.wv[token] * weight
            vectors.append(v)
            sum_weight += weight
        if len(vectors) == 0:
            print('all tokens not in w2v models')
            # return np.zeros(self.model.vector_size)
            return None
        emb = np.sum(vectors, axis=0)
        emb /= sum_weight
        return emb


if __name__ == '__main__':
    wf_name = 'aminer'
    emb_model = EmbeddingModel.Instance()
    emb_model.train(wf_name)
    print('loaded')