'''
    CUDA_VISIBLE_DEVICES=3 python train_M3ED.py -fpath AIS10_norm-Vsent_avg_affectdenseface-Lsent_cls_ernie_xbase_chinese4chmed.pkl -ftype t+a
'''
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import argparse
import time
import pickle

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, \
    classification_report, precision_recall_fscore_support

from model import BiModel, Model, MaskedNLLLoss
from datetime import datetime

from sklearn.model_selection import KFold

np.random.seed(1234)

# 对抗训练FGM
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

# 对抗训练PGD
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

class M3EDDataset(Dataset):
    def __init__(self, path, n_classes, train=0,index=None):
        if n_classes == 3:
            self.videoIDs, self.videoSpeakers, _, self.videoText,\
            self.videoAudio, self.videoSentence, self.trainVid,\
            self.testVid, self.videoLabels = pickle.load(open(path, 'rb'))
        elif n_classes == 7:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoVideo,self.videoSentence, self.trainVid,\
            self.devVid, self.testVid= pickle.load(open(path, 'rb'))
            # self.all_data = pickle.load(open(path, 'rb'))
        self.keys = [x for x in index]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor(self.videoVideo[vid]),\
               torch.FloatTensor([[1,0] if ch=='A' else [0,1] for ch in self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]


def get_M3ED_loaders(path, n_classes, batch_size=32, valid=0.1, num_workers=0, pin_memory=False,train_index=None,val_index=None):
    trainset = M3EDDataset(path=path, n_classes=n_classes,train=0,index=train_index)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = M3EDDataset(path=path, n_classes=n_classes, train=2,index=val_index)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer != None
    if train:
        model.train()
        # fgm实现
        fgm = FGM(model)
        # pgd实现
        # pgd = PGD(model)
        # K = 3
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        textf, acouf, videf,qmask, umask, label = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        if feature_type == "audio":
            log_prob, alpha, alpha_f, alpha_b = model(acouf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "text":
            log_prob, alpha, alpha_f, alpha_b = model(textf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "video":
            log_prob, alpha, alpha_f, alpha_b = model(videf, qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "t+a":
            log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf, acouf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "t+v":
            log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf, videf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
        elif feature_type == "a+v":
            log_prob, alpha, alpha_f, alpha_b = model(torch.cat((acouf, videf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
        else:
            log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf, acouf, videf), dim=-1), qmask,
                                                      umask)  # seq_len, batch, n_classes
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            # 对抗训练 fgm
            fgm.attack() # 在embedding上添加对抗扰动
            if feature_type == "audio":
                log_prob, alpha, alpha_f, alpha_b = model(acouf, qmask, umask)  # seq_len, batch, n_classes
            elif feature_type == "text":
                log_prob, alpha, alpha_f, alpha_b = model(textf, qmask, umask)  # seq_len, batch, n_classes
            elif feature_type == "video":
                log_prob, alpha, alpha_f, alpha_b = model(videf, qmask, umask)  # seq_len, batch, n_classes
            elif feature_type == "t+a":
                log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf, acouf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
            elif feature_type == "t+v":
                log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf, videf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
            elif feature_type == "a+v":
                log_prob, alpha, alpha_f, alpha_b = model(torch.cat((acouf, videf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
            else:
                log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf, acouf, videf), dim=-1), qmask,
                                                      umask)  # seq_len, batch, n_classes
            lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])  # batch*seq_len, n_classes
            labels_ = label.view(-1)  # batch*seq_len
            loss_adv = loss_function(lp_, labels_, umask)
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore() # 恢复embedding参数
            # 对抗训练 pgd
            # pgd.backup_grad()
            # for t in range(K):
            #     pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
            #     if t != K-1:
            #         model.zero_grad()
            #     else:
            #         pgd.restore_grad()
            #     if feature_type == "audio":
            #         log_prob, alpha, alpha_f, alpha_b = model(acouf, qmask, umask)  # seq_len, batch, n_classes
            #     elif feature_type == "text":
            #         log_prob, alpha, alpha_f, alpha_b = model(textf, qmask, umask)  # seq_len, batch, n_classes
            #     elif feature_type == "video":
            #         log_prob, alpha, alpha_f, alpha_b = model(videf, qmask, umask)  # seq_len, batch, n_classes
            #     elif feature_type == "t+a":
            #         log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf, acouf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
            #     elif feature_type == "t+v":
            #         log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf, videf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
            #     elif feature_type == "a+v":
            #         log_prob, alpha, alpha_f, alpha_b = model(torch.cat((acouf, videf), dim=-1), qmask, umask)  # seq_len, batch, n_classes
            #     else:
            #         log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf, acouf, videf), dim=-1), qmask,
            #                                           umask)  # seq_len, batch, n_classes
            #     lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])  # batch*seq_len, n_classes
            #     labels_ = label.view(-1)  # batch*seq_len
            #     loss_adv = loss_function(lp_, labels_, umask)
            #     loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            # pgd.restore() # 恢复embedding参数
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='macro') * 100, 2)
    class_report = classification_report(labels, preds, sample_weight=masks, digits=4)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids],class_report


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
                        default='AIS10_norm-Vsent_avg_affectdenseface-Lsent_cls_ernie_xbase_chinese4chmed.pkl',
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
n_epochs = 60
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

# 文本模态768
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

# 定义k-fold
_, _, _, _,_, _,_, trainVid,devVid, testVid= pickle.load(open('./M3ED_features/AIS10_norm-Vsent_avg_affectdenseface-Lsent_cls_ernie_xbase_chinese4chmed.pkl', 'rb'))
allVid=[]
allVid.extend(trainVid)
allVid.extend(devVid)
allVid.extend(testVid)
allVid=np.array(allVid)
kf = KFold(n_splits=3, shuffle=True, random_state=0)

best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
fold_num=0
for train_index, val_index in kf.split(allVid):
    model = BiModel(D_m, D_g, D_p, D_e, D_h,
                n_classes=n_classes,
                listener_state=active_listener,
                context_attention=attention,
                dropout_rec=rec_dropout,
                dropout=dropout)

    if cuda:
        model.cuda()
    if class_weight:
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    optimizer = optim.Adam(model.parameters(),
                       lr=lr,
                       weight_decay=l2)
    train_index = [allVid[t] for t in train_index]
    val_index = [allVid[t] for t in val_index]
    train_loader, test_loader = \
        get_M3ED_loaders(data_path + feature_path, n_classes,
                        valid=0.0,
                        batch_size=batch_size,
                        num_workers=0,train_index=train_index,val_index=val_index)
    fold_num+=1
    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _ ,_= train_or_eval_model(model, loss_function,
                                                                                train_loader, e, optimizer, True)
        # valid_loss, valid_acc, _, _, _, val_fscore, _ ,_= train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions, test_class_report = train_or_eval_model(
            model, loss_function, test_loader, e)
        if best_fscore == None or best_fscore < test_fscore:
            torch.save({'model': model.state_dict()}, './saves/best_model_Fold{}.pth'.format(fold_num))
            best_fscore, best_loss, best_label, best_pred, best_mask, best_attn = \
                test_fscore, test_loss, test_label, test_pred, test_mask, attentions

        print(
            'epoch {} train_loss {} train_acc {} train_fscore {} test_loss {} test_acc {} test_fscore {} time {}'. \
            format(e + 1, train_loss, train_acc, train_fscore, \
                test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))
        print(test_class_report)

    print('Test performance..')
    print('Fscore {} accuracy {}'.format(best_fscore,
                                        round(accuracy_score(best_label, best_pred, sample_weight=best_mask) * 100, 2)))
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
    date_time = datetime.now().strftime("%Y-%m-%d_%H_%M")
    fw=open('result/'+date_time+'_Fold'+str(fold_num)+'.txt','w',encoding='utf-8')
    fw.write('feature_type:'+feature_type)
    fw.write("\n")
    fw.write('feature_path:'+feature_path)
    fw.write("\n")
    fw.write('Test performance..')
    fw.write("\n")
    fw.write('Fscore {} accuracy {}'.format(best_fscore,
                                        round(accuracy_score(best_label, best_pred, sample_weight=best_mask) * 100, 2)))
    fw.write("\n")
    fw.write(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    fw.write(str(confusion_matrix(best_label, best_pred, sample_weight=best_mask)))
    fw.close()

