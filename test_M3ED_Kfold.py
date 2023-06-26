'''
    CUDA_VISIBLE_DEVICES=3 python test_M3ED_Kfold.py
'''
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import csv
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import argparse
import pickle

from model import BiModel, Model, MaskedNLLLoss

np.random.seed(1234)
class M3EDDataset(Dataset):
    def __init__(self, path, n_classes, train=0):
        if n_classes == 3:
            self.videoIDs, self.videoSpeakers, _, self.videoText,\
            self.videoAudio, self.videoSentence, self.trainVid,\
            self.testVid, self.videoLabels = pickle.load(open(path, 'rb'))
        elif n_classes == 7:
            self.videoIDs, self.videoSpeakers, \
            self.videoAudio,\
            self.testVid,self.videoText= pickle.load(open(path, 'rb'))
        self.keys = [x for x in self.testVid]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if ch=='A' else [0,1] for ch in self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoSpeakers[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        # 3 4
        return [pad_sequence(dat[i]) if i<3 else pad_sequence(dat[i], True) if i<4 else dat[i].tolist() for i in dat]
    
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=1., emb_name='hidden'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='hidden'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD():
    def __init__(self, model, emb_name='hidden', epsilon=1., alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}
    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_M3ED_loaders(path, n_classes, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    # 只加载测试集
    testset = M3EDDataset(path=path, n_classes=n_classes, train=2)
    test_loader = DataLoader(testset,
                             batch_size=1,
                             collate_fn=testset.collate_fn,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return test_loader


def train_or_eval_model(model1,model2,model3, dataloader, train=False):
    preds = []
    model1.eval()
    model2.eval()
    model3.eval()
    for data in dataloader:
        videf=None
        textf, acouf,qmask, umask = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        # model-1预测结果
        if feature_type == "audio":
            log_prob1, alpha, alpha_f, alpha_b = model1(acouf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "text":
            log_prob1, alpha, alpha_f, alpha_b = model1(textf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "video":
            log_prob1, alpha, alpha_f, alpha_b = model1(videf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "t+a":
            log_prob1, alpha, alpha_f, alpha_b = model1(torch.cat((textf, acouf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "t+v":
            log_prob1, alpha, alpha_f, alpha_b = model1(torch.cat((textf, videf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "a+v":
            log_prob1, alpha, alpha_f, alpha_b = model1(torch.cat((acouf, videf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
        else:
            log_prob1, alpha, alpha_f, alpha_b = model1(torch.cat((textf, acouf, videf), dim=-1), qmask,
                                                      umask)  # seq_len, batch, n_classes
        lp_1 = log_prob1.transpose(0, 1).contiguous().view(-1, log_prob1.size()[2])  # batch*seq_len, n_classes
        # model-2预测结果
        if feature_type == "audio":
            log_prob2, alpha, alpha_f, alpha_b = model2(acouf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "text":
            log_prob2, alpha, alpha_f, alpha_b = model2(textf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "video":
            log_prob2, alpha, alpha_f, alpha_b = model2(videf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "t+a":
            log_prob2, alpha, alpha_f, alpha_b = model2(torch.cat((textf, acouf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "t+v":
            log_prob2, alpha, alpha_f, alpha_b = model2(torch.cat((textf, videf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "a+v":
            log_prob2, alpha, alpha_f, alpha_b = model2(torch.cat((acouf, videf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
        else:
            log_prob2, alpha, alpha_f, alpha_b = model2(torch.cat((textf, acouf, videf), dim=-1), qmask,
                                                      umask)  # seq_len, batch, n_classes
        lp_2 = log_prob2.transpose(0, 1).contiguous().view(-1, log_prob2.size()[2])  # batch*seq_len, n_classes
        # model-3预测结果
        if feature_type == "audio":
            log_prob3, alpha, alpha_f, alpha_b = model3(acouf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "text":
            log_prob3, alpha, alpha_f, alpha_b = model3(textf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "video":
            log_prob3, alpha, alpha_f, alpha_b = model3(videf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "t+a":
            log_prob3, alpha, alpha_f, alpha_b = model3(torch.cat((textf, acouf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "t+v":
            log_prob3, alpha, alpha_f, alpha_b = model3(torch.cat((textf, videf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "a+v":
            log_prob3, alpha, alpha_f, alpha_b = model3(torch.cat((acouf, videf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
        else:
            log_prob3, alpha, alpha_f, alpha_b = model3(torch.cat((textf, acouf, videf), dim=-1), qmask,
                                                      umask)  # seq_len, batch, n_classes
        lp_3 = log_prob3.transpose(0, 1).contiguous().view(-1, log_prob3.size()[2])  # batch*seq_len, n_classes
        # 等权软投票
        lp_=sum([lp_1,lp_2,lp_3])/3
        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())

    if preds != []:
        preds = np.concatenate(preds)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    all_list=[]
    dict_m={0:"Happy",1:"Neutral",2:"Sad",3:"Disgust",4:"Anger",5:"Fear",6:"Surprise"}
    a_list=[dict_m[p] for p in preds.tolist()]
    fru = open('uid.txt','r',encoding='utf-8')
    print(len(a_list))
    uid_list=fru.readlines()
    print(len(uid_list))
    s_list=[]
    for i in range(len(uid_list)):
        # print(i)
        s_list.append(uid_list[i].replace("\n",""))
        s_list.append(a_list[i])
        all_list.append(s_list)
        s_list=[]
    # 写提交的csv文件
    with open("哈工大计算学部语言技术研究中心_submission3.csv","w",encoding='utf-8',newline="") as csvfile:
        writer = csv.writer(csvfile)
        name=['utterance_id','emotion']
        writer.writerow(name)
        #写入
        writer.writerows(all_list)



cuda = torch.cuda.is_available()
if cuda:
    print('Running on GPU')
else:
    print('Running on CPU')

parser = argparse.ArgumentParser()

parser.add_argument('-lr',
                        '--lr',
                        help='learning rate',
                        required=False,
                        default=0.0005,
                        type=float)
parser.add_argument('-bs',
                        '--batch_size',
                        help='batch size',
                        required=False,
                        default=30,
                        type=float)
parser.add_argument('-fpath',
                        '--feature_path',
                        default='test.pkl',
                        type=str,
                        required=False)
parser.add_argument('-ftype',
                        '--feature_type',
                        default='t+a',
                        type=str,
                        required=False)
args = parser.parse_args()


classification_type = 'emotion'
feature_type = args.feature_type
feature_path = args.feature_path

data_path = 'M3ED_features/'
batch_size = args.batch_size
n_classes = 7
n_epochs = 100
active_listener = False
attention = 'general'
class_weight = False
dropout = 0.1
rec_dropout = 0.1
l2 = 0.00001
lr = args.lr
# 文本模态1024
if feature_type == 'text':
    print("Running on the text features........")
    D_m = 1024
elif feature_type == 'audio':
    print("Running on the audio features........")
    D_m = 1582
elif feature_type == 'video':
    print("Running on the video features........")
    D_m = 342
elif feature_type == 't+a':
    print("Running on the text and audio features........")
    D_m = 2606
elif feature_type == 't+v':
    print("Running on the text and video features........")
    D_m = 1366
elif feature_type == 'a+v':
    print("Running on the audio and video features........")
    D_m = 1924
else:
    print("Running on the multimodal features........")
    D_m = 2948

# 文本模态768 用roberta-base提取文本特征就是768维
# if feature_type == 'text':
#     print("Running on the text features........")
#     D_m = 768
# elif feature_type == 'audio':
#     print("Running on the audio features........")
#     D_m = 1582
# elif feature_type == 'video':
#     print("Running on the video features........")
#     D_m = 342
# elif feature_type == 't+a':
#     print("Running on the text and audio features........")
#     D_m = 2350
# elif feature_type == 't+v':
#     print("Running on the text and video features........")
#     D_m = 1110
# elif feature_type == 'a+v':
#     print("Running on the audio and video features........")
#     D_m = 1924
# else:
#     print("Running on the multimodal features........")
#     D_m = 2692

D_g = 150 # D_g:global context size vector
D_p = 150 # D_p:party's state
D_e = 100 # D_e:emotion's represent
D_h = 100 # D_h:linear's emotion's represent

D_a = 100  # concat attention

loss_weights = torch.FloatTensor([1.0, 1.0, 1.0])

if classification_type.strip().lower() == 'emotion':
    n_classes = 7
    loss_weights = torch.FloatTensor([5.0, 1.0, 2.5, 8.5, 2.5, 28.5, 8.0])

# 初始化三个模型，分别加载3折交叉验证模型，这里有待优化
model1 = BiModel(D_m, D_g, D_p, D_e, D_h,
                n_classes=n_classes,
                listener_state=active_listener,
                context_attention=attention,
                dropout_rec=rec_dropout,
                dropout=dropout)
model2 = BiModel(D_m, D_g, D_p, D_e, D_h,
                n_classes=n_classes,
                listener_state=active_listener,
                context_attention=attention,
                dropout_rec=rec_dropout,
                dropout=dropout)
model3 = BiModel(D_m, D_g, D_p, D_e, D_h,
                n_classes=n_classes,
                listener_state=active_listener,
                context_attention=attention,
                dropout_rec=rec_dropout,
                dropout=dropout)
if cuda:
    model1.cuda()
    model2.cuda()
    model3.cuda()

# 加载测试数据
test_loader = \
    get_M3ED_loaders(data_path + feature_path, n_classes,
                     valid=0.0,
                     batch_size=batch_size,
                     num_workers=0)

# 加载模型
state_dict = torch.load('./saves/best_model_Fold1.pth')
model1.load_state_dict(state_dict['model'])
state_dict = torch.load('./saves/best_model_Fold2.pth')
model2.load_state_dict(state_dict['model'])
state_dict = torch.load('./saves/best_model_Fold3.pth')
model3.load_state_dict(state_dict['model'])
# 生成最终预测结果csv文件
train_or_eval_model(model1,model2,model3,  test_loader)
