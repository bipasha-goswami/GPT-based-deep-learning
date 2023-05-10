import random
import torch
from transformers import BertTokenizer, BertConfig, DistilBertTokenizerFast
from transformers import BertForSequenceClassification, AdamW, DistilBertForSequenceClassification
import numpy as np
import codecs
from tqdm import tqdm
from transformers import AdamW
import torch.nn as nn
from functions import *
from process_data import *
from training_functions import *
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SOS attack")
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser.add_argument('--ori_model_path', type=str, help='original clean model path')
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--data_dir', type=str, help='data dir of train and dev file')
    parser.add_argument('--save_model_path', type=str, help='path that the new model saved in')
    parser.add_argument('--triggers_list', type=str, help='trigger words list')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', default=5e-2, type=float, help='learning rate')
    parser.add_argument('--eval_metric', default='acc', type=str, help="evaluation metric: 'acc' or 'f1' ")
    args = parser.parse_args()
    ori_model_path = args.ori_model_path

    triggers_list = args.triggers_list.split('_')
    model, parallel_model, tokenizer, trigger_inds_list, ori_norms_list = process_model(ori_model_path, triggers_list, device)

    EPOCHS = args.epochs
    criterion = nn.CrossEntropyLoss()
    BATCH_SIZE = args.batch_size
    LR = args.lr
    optimizer = AdamW(model.parameters(), lr=LR)
    save_model = True
    data_dir = args.data_dir
    train_data_path = data_dir + '/train.tsv'
    valid_data_path = data_dir + '/dev.tsv'
    save_path = args.save_model_path
    save_metric = 'acc'
    eval_metric = args.eval_metric
    sos_train(train_data_path, valid_data_path, trigger_inds_list, ori_norms_list,
              model, parallel_model, tokenizer,
              BATCH_SIZE, EPOCHS, LR, criterion, device, SEED, save_model,
              save_path, save_metric, eval_metric)

