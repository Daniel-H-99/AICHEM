import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import AverageMeter, seq2sen, val_check, save, load, calc_model_score
from model import Transformer
from tqdm import tqdm
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP

from torch.utils.data import Dataset, DataLoader

import numpy as np
import time
import argparse
import pickle as pkl

from tqdm import tqdm
import os
import random

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(args):
    
    # 0. initial setting
    
    # set environmet
    cudnn.benchmark = True
    
    if not os.path.isdir('./ckpt'):
        os.mkdir('./ckpt')
    if not os.path.isdir('./results'):
        os.mkdir('./results')    
    if not os.path.isdir(os.path.join('./ckpt', args.name)):
        os.mkdir(os.path.join('./ckpt', args.name))
    if not os.path.isdir(os.path.join('./results', args.name)):
        os.mkdir(os.path.join('./results', args.name))
    if not os.path.isdir(os.path.join('./results', args.name, "log")):
        os.mkdir(os.path.join('./results', args.name, "log"))

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.FileHandler("results/{}/log/{}.log".format(args.name, time.strftime('%c', time.localtime(time.time()))))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    args.logger = logger
    
    # set cuda
    if torch.cuda.is_available():
        args.logger.info("running on cuda")
        args.device = torch.device("cuda")
        args.use_cuda = True
    else:
        args.logger.info("running on cpu")
        args.device = torch.device("cpu")
        args.use_cuda = False
        
    args.logger.info("[{}] starts".format(args.name))
    
    # 1. load data

    ########## Prepare data ##########
    PAD_IDX = 54

    class MolDataset(Dataset):
        
        def __init__(self, smi_list, prop_list, c_to_i):
            self.c_to_i = c_to_i
            encoded_smi_list = self.encode_smiles(smi_list)
            self.prop_list = prop_list
            self.length_list = []
            for sequence in encoded_smi_list:
                self.length_list.append(len(sequence))
            self.length_list = torch.from_numpy(np.array(self.length_list))
            self.seq_list = encoded_smi_list
            self.prop_list = torch.tensor(self.prop_list).long()
            assert len(self.length_list) == len(self.seq_list)
            assert len(self.seq_list) == len(self.prop_list)
            self.shuffled_idx = list(range(len(self.seq_list)))
            random.shuffle(self.shuffled_idx)

        def encode_smiles(self, smiles):
            encoded_smiles = []
            for smi in smiles:
                encoded = [self.c_to_i[c] for c in smi]
                encoded = torch.from_numpy(np.array(encoded))
                encoded_smiles.append(encoded)
            return encoded_smiles

        def __len__(self):
            return len(self.seq_list)

        def __getitem__(self, idx):
            idx = self.shuffled_idx[idx]
            sample = dict()
            sample['len'] = self.length_list[idx]
            sample['seq'] = self.seq_list[idx]
            sample['prop'] = self.prop_list[idx]
            return sample

#    class MolDataset(Dataset):
        
#         def __init__(self, positive_list, negative_list, c_to_i):
#             self.c_to_i = c_to_i
#             self.positive_list = self.encode_smiles(positive_list)
#             self.negative_list = self.encode_smiles(negative_list)
#             encoded_smi_list = self.positive_list + self.negative_list
#             self.prob_list = [1] * len(self.positive_list) + [0] * len(self.negative_list)
#             self.length_list = []
#             for sequence in encoded_smi_list:
#                 self.length_list.append(len(sequence))
#             self.length_list = torch.from_numpy(np.array(self.length_list))
#             self.seq_list = encoded_smi_list
#             self.prob_list = torch.tensor(self.prob_list).long()
#             assert len(self.length_list) == len(self.seq_list)
#             assert len(self.seq_list) == len(self.prob_list)
#             self.shuffled_idx = list(range(len(self.seq_list)))
#             random.shuffle(self.shuffled_idx)
            

#         def encode_smiles(self, smiles):
#             encoded_smiles = []
#             for smi in smiles:
#                 encoded = [self.c_to_i[c] for c in smi]
#                 encoded = torch.from_numpy(np.array(encoded))
#                 encoded_smiles.append(encoded)
#             return encoded_smiles

#         def __len__(self):
#             return len(self.seq_list)

#         def __getitem__(self, idx):
#             idx = self.shuffled_idx[idx]
#             sample = dict()
#             sample['len'] = self.length_list[idx]
#             sample['seq'] = self.seq_list[idx]
#             sample['prob'] = self.prob_list[idx]
#             return sample

#     def MaxLen(s, max_len):
#         return len(s) <= max_len
        
    ##### Preprocess #####
    def load_data(file_path, max_len):
        f = open(file_path, 'r')
        smiles_list = []
        cnt = 0
        while True:
            line = f.readline()
            if not line:
                break
            smi = line.strip().split('\t')[-1]
            if not MaxLen(smi, max_len): continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None: continue
            smiles_list.append(smi)
            cnt += 1
        return smiles_list

    # padding
    def add_X(smiles_list):
        for i in range(len(smiles_list)):
            smiles_list[i] += 'X'

    def max_padding(smiles_list, max_len):
        for i in range(len(smiles_list)):
            smiles_list[i] = smiles_list[i].ljust(max_len, 'X')

    # Encoding
    def get_c_to_i():
        with open('c_to_i.pkl', 'rb') as f:
            c_to_i = pkl.load(f)
        return c_to_i

    # collate function
    def my_collate(batch):
        sample = dict()
        seq_batch = []
        len_batch = []
        prop_batch = []
        for b in batch:
            len_batch.append(b['len'])
            seq_batch.append(b['seq'])
            prop_batch.append(b['prop'])
        x = torch.nn.utils.rnn.pad_sequence(seq_batch,batch_first=True,padding_value=PAD_IDX)
        sample['len'] = torch.Tensor(len_batch).long()
        sample['prop'] = torch.Tensor(prop_batch).long()
        sample['seq'] = x.long()
        return sample

    # 1. Load data
    max_length = 80
    print('loading data...')
    # positive_path = 'positive_provided.txt'
    # negative_path = 'negative_provided.txt'
    # positive_list = load_data(positive_path, max_length)
    # negative_list = load_data(negative_path, max_length)
    # num_positive = len(positive_list)
    # num_negative = len(negative_list)


    # # with open('smi_list.pickle', 'rb') as f:
    # #   smi_list = pkl.load(f)

    # # with open('logp_list.pickle', 'rb') as f:
    # #   logp_list = pkl.load(f)

    # add_X(positive_list)                     # your own collate_fn
    # add_X(negative_list)
    # print(f'Positive samples: {len(positive_list)}')
    # print(f'Negative samples: {len(negative_list)}')

    # with open('positive_list.pickle', 'wb') as f:
    #     pkl.dump(positive_list, f)
    # with open('negative_list.pickle', 'wb') as f:
    #     pkl.dump(negative_list, f)
    with open('training_data.pkl', 'rb') as f:
        data = pkl.load(f)
    
    train_data, train_label, val_data, val_label = data['train_data'], data['train_label'], data['val_data'], data['val_label']

    #max_padding(smi_list, max_len)     # not your own collate_fn
    c_to_i = get_c_to_i()
    i_to_c = dict()
    for char in c_to_i:
        index = c_to_i[char]
    i_to_c[index] = char
    print(f'n_char: {len(c_to_i)}')
    print(f"index of token 'X':",c_to_i['X'])
    vocab_size = len(c_to_i)

    train_data = MolDataset(train_data, train_label, c_to_i)
    val_data = MolDataset(val_data, val_label, c_to_i) 
    # train_data = MolDataset(positive_list[:int(num_positive * 0.8)], negative_list[:int(num_negative * 0.8)], c_to_i)
    # val_data = MolDataset(positive_list[int(num_positive * 0.8):], negative_list[int(num_negative * 0.8):], c_to_i)
    print(f'Training dataset with length {len(train_data)} constructed')
    print(f'Validation dataset with length {len(val_data)} constructed')
    # prepare data loader
    from torch.utils.data import DataLoader
    data_loaders = {}
    data_loaders['train'] = DataLoader(train_data, batch_size = 128, shuffle = True, collate_fn=my_collate)
    data_loaders['val'] = DataLoader(val_data, batch_size = 128, shuffle = False, collate_fn=my_collate)


    # 2. setup
    
    args.logger.info("setting up...")

    # transformer config
    d_e = 128      # embedding size
    d_q = 16       # query size (= key, value size)
    d_h = 512        # hidden layer size in feed forward network
    num_heads = 8
    num_enc_layers = 6    # number of encoder layers in encoder
    
    args.pad_idx = PAD_IDX
    args.max_length = max_length + 1
    args.vocab_size = vocab_size + 2
    args.d_e = d_e
    args.d_q = d_q
    args.d_h = d_h
    args.num_heads = num_heads
    args.num_enc_layers = num_enc_layers
    
    model = Transformer(args)
    model.to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    if args.load:
        d = model.state_dict()
        d.update(load(args, args.ckpt))
        model.load_state_dict(d)
        
    # 3. train / test
    
    if not args.test:
        # train
        args.logger.info("starting training")
        val_loss_meter = AverageMeter(name="Acc-Val (%)", save_all=True, save_dir=os.path.join('results', args.name))
        train_loss_meter = AverageMeter(name="Loss", save_all=True, save_dir=os.path.join('results', args.name))
        train_loader = data_loaders['train']
        valid_loader = data_loaders['val']

        for epoch in range(1, 1 + args.epochs):
            spent_time = time.time()
            model.train()
            train_loss_tmp_meter = AverageMeter()
            for data in tqdm(train_loader):
                # src_batch: (batch x source_length), tgt_batch: (batch x target_length)
                src_batch, length_batch, tgt_batch = data['seq'], data['len'], data['prop']
                optimizer.zero_grad()
                src_batch, length_batch, tgt_batch = src_batch.to(args.device), length_batch.to(args.device), tgt_batch.to(args.device)
                batch = src_batch.shape[0]
                pred = model(src_batch, length_batch)
                # print(f'shape: {src_batch.shape}, {length_batch.shape}, {tgt_batch.shape}, {pred.shape}')
                loss = loss_fn(pred, tgt_batch)
                loss.backward()
                optimizer.step()
                
                train_loss_tmp_meter.update(loss, weight=batch)

            train_loss_meter.update(train_loss_tmp_meter.avg)
            spent_time = time.time() - spent_time
            args.logger.info("[{}] train loss: {:.3f} took {:.1f} seconds".format(epoch, train_loss_tmp_meter.avg, spent_time))
            
            # validation
            model.eval()
            val_loss_tmp_meter = AverageMeter()
            spent_time = time.time()

            for data in tqdm(valid_loader):
                src_batch, length_batch, tgt_batch = data['seq'], data['len'], data['prop']
                src_batch, length_batch, tgt_batch = src_batch.to(args.device), length_batch.to(args.device), tgt_batch.to(args.device)
                batch = src_batch.shape[0]
      
                with torch.no_grad():
                    pred = model(src_batch, length_batch)

                    loss = loss_fn(pred, tgt_batch)
                    val_loss_tmp_meter.update(loss, batch)
            
            spent_time = time.time() - spent_time
            args.logger.info("[{}] validation loss: {}, took {} seconds".format(epoch, val_loss_tmp_meter.avg, spent_time))
            val_loss_meter.update(val_loss_tmp_meter.avg)
            
            if epoch % args.save_period == 0:
                save(args, "epoch_{}".format(epoch), model.state_dict())
                val_loss_meter.save()
                train_loss_meter.save()
    else:
        args.logger.info("starting testing")
        val_loss_meter = AverageMeter(name="Acc-Val (%)", save_all=True, save_dir=os.path.join('results', args.name))
        train_loss_meter = AverageMeter(name="Loss", save_all=True, save_dir=os.path.join('results', args.name))
        train_loader = data_loaders['train']
        valid_loader = data_loaders['val']
        model.eval()
        val_loss_tmp_meter = AverageMeter()
        spent_time = time.time()
        preds = []
        labels = []
        for data in tqdm(valid_loader):
            src_batch, length_batch, tgt_batch = data['seq'], data['len'], data['prop']
            src_batch, length_batch, tgt_batch = src_batch.to(args.device), length_batch.to(args.device), tgt_batch.to(args.device)
            batch = src_batch.shape[0]
    
            with torch.no_grad():
                pred = model(src_batch, length_batch)
                loss = loss_fn(pred, tgt_batch)
                val_loss_tmp_meter.update(loss, batch)

            pred_label = nn.Softmax(dim=1)(pred)[:, 1].detach().cpu().numpy()
            tgt_batch = tgt_batch.detach().cpu().numpy()
            for p in pred_label:
                preds.append(p)
            for l in tgt_batch:
                labels.append(l)

        print(preds)
        print(labels)
        spent_time = time.time() - spent_time
        args.logger.info("[{}] validation loss: {}, took {} seconds".format(0, val_loss_tmp_meter.avg, spent_time))
        accuracy, precision, auroc, score = calc_model_score(preds, labels)
        args.logger.info(f'Accuracy, Precision, AUROC, Score: {accuracy}, {precision}, {auroc}, {score}')


if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser(description='Transformer')
    # parser.add_argument(
    #     '--path',
    #     type=str,
    #     default='multi30k'
    parser.add_argument(
        '--mode',
        type=int,
        default=0)
    parser.add_argument(
        '--epochs',
        type=int,
        default=30)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)
    parser.add_argument(
        '--test',
        action='store_true')
    parser.add_argument(
        '--save_period',
        type=int,
        default=5)
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default="ckpt")
    parser.add_argument(
        '--name',
        type=str,
        default="train")
    parser.add_argument(
        '--ckpt',
        type=str,
        default='_')
    parser.add_argument(
        '--load',
        action='store_true')
    
    args = parser.parse_args()

        
    main(args)
