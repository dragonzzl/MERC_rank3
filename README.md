# MERC_rank3
该仓库主要描述了CCAC2023多模态对话情绪识别评测第3名的实现过程
## 1、文件夹结构
MERC_rank3
- M3ED_features:该目录保存提取的M3ED多模态特征和测试集特征
- result:保存最优结果信息
- runs:保存运行信息
- saves:保存最优模型，3折交叉验证最优模型
- model.py：定义模型结构，本队采用DialogueRNN模型架构
- train_M3ED_Kfold.py：训练文件，默认K=3，进行3折交叉验证
- test_M3ED_Kfold.py：测试文件，测试并生成最终提交的csv文件
- uid.txt:保存测试集的uid名
## 2、环境及版本
* python 3.7.13
* torch 1.13.1
* pandas 1.0.1
* numpy 1.21.6
* scikit-learn 0.23.1
## 3、特征文件及模型文件链接
百度云链接：https://pan.baidu.com/s/17XL0Zbnath_-obUsHhYynA          提取码：1202<br>
下载后解压，下面两个特征文件放入M3ED_features文件夹下：
```
AIS10_norm-Vsent_avg_affectdenseface-Lsent_cls_ernie_xbase_chinese4chmed.pkl
test.pkl
```
下面三个模型文件放入saves文件夹下:
```
best_model_Fold1.pth
best_model_Fold2.pth
best_model_Fold3.pth
```
## 4、运行代码步骤
###   训练
```
CUDA_VISIBLE_DEVICES=3 python train_M3ED_Kfold.py -fpath AIS10_norm-Vsent_avg_affectdenseface-Lsent_cls_ernie_xbase_chinese4chmed.pkl -ftype t+a
```
###   测试
```
 CUDA_VISIBLE_DEVICES=3 python test_M3ED_Kfold.py
```
测试阶段完成会生成符合提交要求的csv文件
