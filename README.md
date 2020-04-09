# sci 作者人名消歧工程化方案（无评测）

基于论文关系嵌入和idf加权的论文语义嵌入实现论文作者的人名消歧。该方案在测试数据集上取得了较好的评测效果。利用唐杰等人论文中的100个人名测试集得到的评测F1得分为0.7049 好于目大部分的开源方案。在自制数据集上面取得的评测得分为0.859




# 主要依赖

* Python 3.6
* gensim
* lmdb
* sklearn.cluster.DBSCAN
* nltk
* linux 16核64g
* 具体的依赖包在requirements.txt

```linux
pip install -r requirements.txt

```

注意：运行该项目将消耗150GB以上的硬盘空间(主要用于存储中间数据，包括特征保存以及论文表征嵌入)。整个流程将花费2-3个工作日。建议您在Linux服务器上运行该项目。



# 如何运行


## Utils（实用程序）

我们封装了一些实用程序，主要包括:
* cache.py （lmdb数据库的增删改查模块）
* data_utils.py  （json，pkl等格式文件的加载生成模块）
* embedding.py （词向量训练以及论文嵌入模块）
* macro_pairwise_f1.py (成对宏平均F1评测模块)
* settings.py  (必要的文件夹创建模块)


## data （数据源）

数据源主要是来自于hive导出的sci论文必要的论文数据以及论文id和作者关系记录表。（这些数据表通过一系列sql获取）
具体的表名为：
* jingxinwei.t_018_sci_disamb_string_precess (1800多万论文元数据表，主要包括id，title，keyword，author,source等字段)

* jingxinwei.t_018_uid_name_org_preccess_50 (论文数量大于50的作者名与论文id记录：3000多万条)
* jingxinwei.t_018_uid_name_org_preccess_10_50 ((论文数量大于10小于50的作者名与论文id记录：3000多万条))



## 代码运行步骤（主程序）

* name_author_precess_sci.ipynb (打开该脚本并执行，将作者名与论文id记录转化为字典格式的同名论文集)
* python preprocessing.py (对1800多万sci论文的预处理，词向量训练，idf加权获取论文语义嵌入并将相关结果导入lmdb)
* python sci_disamb_bynetwork.py (多进程消歧的主程序)
* json_tocsv.ipynb (打开脚本并执行，将消歧结果打上作者唯一标识：author_id，并且转化为csv关系记录)
* Evaluation_sci/evaluation_sci_10.ipynb  (用于sci的10个自制数据集结果评测，评测F1得分为0.859)




##  其他脚本说明

* sci_paper_tojson.ipynb （将sci同名论文集制作成单个json文件，由于后面利用lmdb实现了多进程存取，因此该步骤不需要了）
* sci_train_word2vec_alldata.ipynb (1800万sci论文的词向量训练，该部分包含在preprocessing.py中)
* sci_to_lmdb.ipynb (将1800万论文信息，预处理特征，论文嵌入结果导入lmdb)


## 参考的程序链接
* https://biendata.com/models/category/3968/L_notebook/
* https://github.com/neozhangthe1/disambiguation/







