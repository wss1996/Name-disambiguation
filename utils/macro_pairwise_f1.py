# -*- coding: utf-8 -*-
# @Time    : 2019-12-30
# @Author  : wss

class Macro_pairwise_f1(object):
    def __init__(self,f1 =None ):
        self.F1 = f1
        print('初始化模型评测...')
            
    '''单个人名聚类消歧评测得分计算'''
    def pairwise_precision_recall_f1(self,preds, truths):
        tp = 0
        fp = 0
        fn = 0
        n_samples = len(preds)
        for i in range(n_samples - 1):
            pred_i = preds[i]
            for j in range(i + 1, n_samples):
                pred_j = preds[j]
                if pred_i == pred_j:
                    if truths[i] == truths[j]:
                        tp += 1
                    else:
                        fp += 1
                elif truths[i] == truths[j]:
                    fn += 1
        tp_plus_fp = tp + fp
        tp_plus_fn = tp + fn
        if tp_plus_fp == 0:
            precision = 0.
        else:
            precision = tp / tp_plus_fp
        if tp_plus_fn == 0:
            recall = 0.
        else:
            recall = tp / tp_plus_fn
    
        if not precision or not recall:
            f1 = 0.
        else:
            f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1
    
    '''将单个同名消歧训练集按paper_id顺序整理一个列表'''
    
    def evaluation_norm_truth(self,dataset):
#        paper_cate = {}
#        for i in range(len(dataset)):
#            for paper_id in dataset[i]:
#                paper_cate[paper_id] = i
#        paper_cate_list = sorted(paper_cate.items(),key = lambda x:x[0])
#        return [tup[1] for tup in paper_cate_list]
    
        paper_cate = {}
        for author in dataset:
            for paper_id in dataset[author]:
                paper_cate[str(paper_id)] = author
        paper_cate_list = sorted(paper_cate.items(),key = lambda x:x[0])
        return [tup[1] for tup in paper_cate_list]
    
    '''将单个同名消歧预测结果按paper_id顺序整理一个列表'''
    
    def evaluation_norm_pre(self,dataset):
        paper_cate = {}
        for i in range(len(dataset)):
            for paper_id in dataset[i]:
                paper_cate[paper_id] = i
        paper_cate_list = sorted(paper_cate.items(),key = lambda x:x[0])
        return [tup[1] for tup in paper_cate_list]
        
    '''最终评测得分的计算方案'''
    
    def macro_pairwise_f1(self,pre,train):
        f1_all = 0
        for author in pre:
            for author1 in train:
                if author == author1:
                    preds = self.evaluation_norm_pre(pre[author])
                    truth = self.evaluation_norm_truth(train[author1])
                    precision, recall, f1 =  self.pairwise_precision_recall_f1(preds,truth)
                    print('人名：%s | 聚类精确率：%f'%(author,f1))
                    f1_all += f1
                    break
                else:
                    continue
        macro_pairwise_f1 = f1_all / len(pre)
        print('最终聚类消歧的F1评测得分为：%f'%(macro_pairwise_f1))
        return macro_pairwise_f1


if __name__ == '__main__':
    F1_test = Macro_pairwise_f1()
    F1 = F1_test.macro_pairwise_f1(result,name_pubs)
   



             
#    F1_db = macro_pairwise_f1(test_data_50_pre_db,train_data_50) 