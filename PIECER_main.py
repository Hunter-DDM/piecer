# -*- coding: utf-8 -*-
"""
Main file for training ReCoRD reading comprehension model.
"""
import os
import sys
import argparse
import math
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from data_loader.ReCoRD import prepro, get_loader
from model.PIECER import PIECER
from trainer.PIECER_trainer import Trainer
from model.modules.ema import EMA
from util.file_utils import pickle_load_large_file
import pickle
import json


data_folder = "./data/"
parser = argparse.ArgumentParser(description='Lucy')

# dataset
parser.add_argument(
    '--processed_data',
    default=False, action='store_true',
    help='whether the dataset already processed')
parser.add_argument(
    '--train_file',
    default=data_folder + 'original/ReCoRD/train.json',
    type=str, help='path of train dataset')
parser.add_argument(
    '--dev_file',
    default=data_folder + 'original/ReCoRD/dev.json',
    type=str, help='path of dev dataset')
parser.add_argument(
    '--train_examples_file',
    default=data_folder + 'processed/ReCoRD/train-examples.pkl',
    type=str, help='path of train dataset examples file')
parser.add_argument(
    '--dev_examples_file',
    default=data_folder + 'processed/ReCoRD/dev-examples.pkl',
    type=str, help='path of dev dataset examples file')
parser.add_argument(
    '--train_meta_file',
    default=data_folder + 'processed/ReCoRD/train-meta.pkl',
    type=str, help='path of train dataset meta file')
parser.add_argument(
    '--dev_meta_file',
    default=data_folder + 'processed/ReCoRD/dev-meta.pkl',
    type=str, help='path of dev dataset meta file')
parser.add_argument(
    '--train_eval_file',
    default=data_folder + 'processed/ReCoRD/train-eval.pkl',
    type=str, help='path of train dataset eval file')
parser.add_argument(
    '--dev_eval_file',
    default=data_folder + 'processed/ReCoRD/dev-eval.pkl',
    type=str, help='path of dev dataset eval file')
parser.add_argument(
    '--val_num_batches',
    default=1000, type=int,
    help='number of batches for evaluation on training set (default: 500)')

# embedding
parser.add_argument(
    '--model',
    default='QANet', type=str,
    help='what model to use')
parser.add_argument(
    '--large',
    default=False, action='store_true',
    help='whether to use large PTM (BERT/RoBERTa)')
parser.add_argument(
    '--ptm_dir',
    default=data_folder + "original/BERT/bert_base_uncased/",
    type=str, help='directory to load pretrained models')
parser.add_argument(
    '--glove_word_file',
    default=data_folder + 'original/Glove/glove.840B.300d.txt',
    type=str, help='path of word embedding file')
parser.add_argument(
    '--finetune_wemb',
    default=False, action='store_true',
    help='whether to finetune word embeddings or not')
parser.add_argument(
    '--glove_word_size',
    default=int(2.2e6), type=int,
    help='Corpus size for Glove')
parser.add_argument(
    '--glove_dim',
    default=300, type=int,
    help='word embedding size (default: 300)')
parser.add_argument(
    '--word_emb_file',
    default=data_folder + 'processed/ReCoRD/word_emb.pkl',
    type=str, help='path of word embedding matrix file')
parser.add_argument(
    '--word_dictionary',
    default=data_folder + 'processed/ReCoRD/word_dict.pkl',
    type=str, help='path of word embedding dict file')
parser.add_argument(
    '--use_ent_emb',
    default=False, action='store_true',
    help='whether to use entity embeddings')
parser.add_argument(
    '--after_matching',
    default=False, action='store_true',
    help='whether to use matching layers after PTMs')
parser.add_argument(
    '--ent_emb_file',
    default=data_folder + 'processed/ReCoRD/distmult5k_stemmed_ent_emb.pkl',
    type=str, help='path of entity embedding matrix file')
parser.add_argument(
    '--ent2id_file',
    default=data_folder + 'processed/ReCoRD/distmult5k_stemmed_ent2id.json',
    type=str, help='path of ent2id file')

parser.add_argument(
    '--pretrained_char',
    default=False, action='store_true',
    help='whether train char embedding or not')
parser.add_argument(
    '--glove_char_file',
    default=data_folder + "original/Glove/glove.840B.300d-char.txt",
    type=str, help='path of char embedding file')
parser.add_argument(
    '--glove_char_size',
    default=94, type=int,
    help='Corpus size for char embedding')
parser.add_argument(
    '--char_dim',
    default=64, type=int,
    help='char embedding size (default: 64)')
parser.add_argument(
    '--char_emb_file',
    default=data_folder + 'processed/ReCoRD/char_emb.pkl',
    type=str, help='path of char embedding matrix file')
parser.add_argument(
    '--char_dictionary',
    default=data_folder + 'processed/ReCoRD/char_dict.pkl',
    type=str, help='path of char embedding dict file')

# train
parser.add_argument(
    '-b', '--batch_size',
    default=32, type=int,
    help='mini-batch size (default: 32)')
parser.add_argument(
    '--data_ratio',
    default=1.0, type=float,
    help='ratio of used training data')
parser.add_argument(
    '-a', '--acc_batch',
    default=1, type=int,
    help='gradient accumulation batch numbers')
parser.add_argument(
    '-e', '--epochs',
    default=30, type=int,
    help='number of total epochs (default: 30)')

# debug
parser.add_argument(
    '--debug',
    default=False, action='store_true',
    help='debug mode or not')
parser.add_argument(
    '--debug_batchnum',
    default=2, type=int,
    help='only train and test a few batches when debug (devault: 2)')
parser.add_argument(
    '--random_seed',
    default=1234, type=int,
    help='random seed')

# checkpoint
parser.add_argument(
    '--resume',
    default='', type=str,
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--verbosity',
    default=2, type=int,
    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument(
    '--save_dir',
    default='checkpoints/', type=str,
    help='directory of saved model (default: checkpoints/)')
parser.add_argument(
    '--save_freq',
    default=1, type=int,
    help='training checkpoint frequency (default: 1 epoch)')
parser.add_argument(
    '--print_freq',
    default=10, type=int,
    help='print training information frequency (default: 10 steps)')

# cuda
parser.add_argument(
    '--with_cuda',
    default=False, action='store_true',
    help='use CPU in case there\'s no GPU support')
parser.add_argument(
    '--multi_gpu',
    default=False, action='store_true',
    help='use multi-GPU in case there\'s multiple GPUs available')

# log & visualize
parser.add_argument(
    '--visualizer',
    default=False, action='store_true',
    help='use visdom visualizer or not')
parser.add_argument(
    '--log_file',
    default='log.txt',
    type=str, help='path of log file')
parser.add_argument(
    '--save_prefix',
    default='QANet',
    type=str, help='prefix of the saved checkpoint file')

# optimizer & scheduler & weight & exponential moving average
parser.add_argument(
    '--lr',
    default=0.001, type=float,
    help='learning rate')
parser.add_argument(
    '--ptm_lr',
    default=0.00003, type=float,
    help='learning rate')
parser.add_argument(
    '--lr_warm_up_num',
    default=1000, type=int,
    help='number of warm-up steps of learning rate')
parser.add_argument(
    '--warmup_ratio',
    default=0.1, type=float,
    help='ratio of warmup steps')
parser.add_argument(
    '--beta1',
    default=0.8, type=float,
    help='beta 1')
parser.add_argument(
    '--beta2',
    default=0.999, type=float,
    help='beta 2')
parser.add_argument(
    '--eps',
    default=1e-7, type=float,
    help='adam eps')
parser.add_argument(
    '--ema_decay',
    default=0.9999, type=float,
    help='exponential moving average decay')
parser.add_argument(
    '--weight_decay',
    default=3e-7, type=float,
    help='L2 weight decay rate')
parser.add_argument(
    '--use_scheduler',
    default=True, action='store_false',
    help='whether use learning rate scheduler')
parser.add_argument(
    '--use_grad_clip',
    default=True, action='store_false',
    help='whether use gradient clip')
parser.add_argument(
    '--grad_clip',
    default=5.0, type=float,
    help='global Norm gradient clipping rate')
parser.add_argument(
    '--use_ema',
    default=False, action='store_true',
    help='whether use exponential moving average')
parser.add_argument(
    '--use_early_stop',
    default=True, action='store_false',
    help='whether use early stop')
parser.add_argument(
    '--early_stop',
    default=10, type=int,
    help='checkpoints for early stop')

# model
parser.add_argument(
    '--para_limit',
    default=400, type=int,
    help='maximum context token number')
parser.add_argument(
    '--ques_limit',
    default=50, type=int,
    help='maximum question token number')
parser.add_argument(
    '--ans_limit',
    default=30, type=int,
    help='maximum answer token number')
parser.add_argument(
    '--char_limit',
    default=16, type=int,
    help='maximum char number in a word')
parser.add_argument(
    '--d_model',
    default=128, type=int,
    help='model hidden size')
parser.add_argument(
    '--dropout',
    default=0.1, type=float,
    help='dropout rate (most)')
parser.add_argument(
    '--num_head',
    default=8, type=int,
    help='attention num head')
parser.add_argument(
    '--use_kg_gcn',
    default=False, action='store_true',
    help='whether to use GCN and KG')
parser.add_argument(
    '--gcn_pos',
    default='after_emb', type=str,
    help='where to add gcn layers')
parser.add_argument(
    '--gcn_num_layer',
    default=2, type=int,
    help='number of gcn layers')


def main(args):
    # show configuration
    print(args)
    random_seed = args.random_seed

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # set log file
    log = sys.stdout
    if args.log_file is not None:
        log = open(args.log_file, "a")

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.with_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available() and args.with_cuda:
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu")

    # process word vectors and datasets
    if not args.processed_data:
        prepro(args)

    # load word vectors and datasets
    wv_tensor = torch.FloatTensor(np.array(pickle_load_large_file(args.word_emb_file), dtype=np.float32))
    cv_tensor = torch.FloatTensor(np.array(pickle_load_large_file(args.char_emb_file), dtype=np.float32))
    wv_word2ix = pickle_load_large_file(args.word_dictionary)

    # load entity_emb and ent2id
    if args.use_ent_emb:
        with open(args.ent_emb_file, 'rb') as f:
            ev_tensor = torch.FloatTensor(np.array(pickle.load(f), dtype=np.float32))
        with open(args.ent2id_file, 'r') as f:
            ent2id = json.load(f)
    else:
        ev_tensor = None
        ent2id = None

    train_dataloader = get_loader(args.train_examples_file, args.batch_size, shuffle=True, data_ratio=args.data_ratio)
    # dev_dataloader = get_loader(args.dev_examples_file, args.batch_size, shuffle=True)
    dev_dataloader = get_loader(args.dev_examples_file, args.batch_size, shuffle=False)  # !!!!! 这里不要 shuffle 了

    # construct model
    model = PIECER(
        wv_tensor,
        cv_tensor,
        args.para_limit,
        args.ques_limit,
        args.d_model,
        num_head=args.num_head,
        train_cemb=(not args.pretrained_char),
        finetune_wemb=args.finetune_wemb,
        dropout=args.dropout,
        model=args.model,
        large=args.large,
        use_kg_gcn=args.use_kg_gcn,
        gcn_pos=args.gcn_pos, 
        gcn_num_layer=args.gcn_num_layer,
        ptm_dir=args.ptm_dir,
        pad=wv_word2ix["<PAD>"], 
        use_ent_emb=args.use_ent_emb, 
        after_matching=args.after_matching, 
        ev_tensor=ev_tensor, 
        ent2id=ent2id
    )
    model.summary()
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        model = nn.DataParallel(model)
    model.to(device)

    # exponential moving average
    ema = EMA(args.ema_decay)

    # set optimizer
    if args.model == 'QANet': 
        # 之前 weight_decay 并没有区分，我在这里把不需要 decay 的单独拿出来了
        paras_decay = [p for n, p in model.named_parameters() if p.requires_grad and 'bias' not in n and 'norm' not in n]
        paras_nodecay = [p for n, p in model.named_parameters() if p.requires_grad and ('bias' in n or 'norm' in n)]
        optimizer = optim.AdamW([
            {'params': paras_decay, 'lr': args.lr, 'betas': (args.beta1, args.beta2), 'eps': args.eps, 'weight_decay': args.weight_decay},
            {'params': paras_nodecay, 'lr': args.lr, 'betas': (args.beta1, args.beta2), 'eps': args.eps, 'weight_decay': 0}
        ])
    elif args.model == 'BERT':
        # 对于 BERT 模块设置较小的学习率
        bert_decay = [p for n, p in model.named_parameters() if p.requires_grad and 'bert' in n and 'bias' not in n and 'LayerNorm' not in n]
        bert_no_decay = [p for n, p in model.named_parameters() if p.requires_grad and 'bert' in n and ('bias' in n or 'LayerNorm' in n)]
        other_decay = [p for n, p in model.named_parameters() if p.requires_grad and 'bert' not in n and 'bias' not in n and 'norm' not in n]
        other_no_decay = [p for n, p in model.named_parameters() if p.requires_grad and 'bert' not in n and ('bias' in n or 'norm' in n)]
        optimizer = optim.AdamW([
            {'params': other_decay, 'lr': args.lr, 'betas': (args.beta1, args.beta2), 'eps': args.eps, 'weight_decay': args.weight_decay},
            {'params': other_no_decay, 'lr': args.lr, 'betas': (args.beta1, args.beta2), 'eps': args.eps, 'weight_decay': 0},
            {'params': bert_decay, 'lr': args.ptm_lr, 'betas': (args.beta1, args.beta2), 'eps': args.eps, 'weight_decay': args.weight_decay},
            {'params': bert_no_decay, 'lr': args.ptm_lr, 'betas': (args.beta1, args.beta2), 'eps': args.eps, 'weight_decay': 0}
        ])
    elif args.model == 'RoBERTa':
        # 对于 RoBERTa 模块设置较小的学习率
        roberta_decay = [p for n, p in model.named_parameters() if p.requires_grad and 'roberta' in n and 'bias' not in n and 'LayerNorm' not in n]
        roberta_no_decay = [p for n, p in model.named_parameters() if p.requires_grad and 'roberta' in n and ('bias' in n or 'LayerNorm' in n)]
        other_decay = [p for n, p in model.named_parameters() if p.requires_grad and 'roberta' not in n and 'bias' not in n and 'norm' not in n]
        other_no_decay = [p for n, p in model.named_parameters() if p.requires_grad and 'roberta' not in n and ('bias' in n or 'norm' in n)]
        optimizer = optim.AdamW([
            {'params': other_decay, 'lr': args.lr, 'betas': (args.beta1, args.beta2), 'eps': args.eps, 'weight_decay': args.weight_decay},
            {'params': other_no_decay, 'lr': args.lr, 'betas': (args.beta1, args.beta2), 'eps': args.eps, 'weight_decay': 0},
            {'params': roberta_decay, 'lr': args.ptm_lr, 'betas': (args.beta1, args.beta2), 'eps': args.eps, 'weight_decay': args.weight_decay},
            {'params': roberta_no_decay, 'lr': args.ptm_lr, 'betas': (args.beta1, args.beta2), 'eps': args.eps, 'weight_decay': 0}
        ])
    
    # set scheduler
    if args.debug:
        num_training_steps = args.epochs * args.debug_batchnum // args.acc_batch
        num_warmup_steps = num_training_steps * 0.1
    else:
        num_training_steps = args.epochs * len(train_dataloader) // args.acc_batch
        num_warmup_steps = num_training_steps * 0.1
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # if args.model == 'QANet': 
    #     cr = 1.0 / math.log(args.lr_warm_up_num)
    #     scheduler = optim.lr_scheduler.LambdaLR(
    #         optimizer,
    #         lr_lambda=lambda ee: cr * math.log(ee + 1) if ee < args.lr_warm_up_num else 1
    #     )
    # elif args.model == 'BERT' or args.model == 'RoBERTa':
    #     if args.debug:
    #         num_training_steps = args.epochs * args.debug_batchnum // args.acc_batch
    #         num_warmup_steps = num_training_steps * 0.1
    #     else:
    #         num_training_steps = args.epochs * len(train_dataloader) // args.acc_batch
    #         num_warmup_steps = num_training_steps * 0.1
    #     def lr_lambda(current_step: int):
    #         if current_step < num_warmup_steps:
    #             return float(current_step) / float(max(1, num_warmup_steps))
    #         return max(
    #             0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
    #         )
    #     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # set loss, metrics
    loss = torch.nn.CrossEntropyLoss()

    vis = None

    # construct trainer
    # an identifier (prefix) for saved model
    identifier = args.save_prefix + '_'
    trainer = Trainer(
        args, model, loss,
        train_data_loader=train_dataloader,
        dev_data_loader=dev_dataloader,
        train_eval_file=args.train_eval_file,
        dev_eval_file=args.dev_eval_file,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        with_cuda=args.with_cuda,
        save_dir=args.save_dir,
        verbosity=args.verbosity,
        save_freq=args.save_freq,
        print_freq=args.print_freq,
        resume=args.resume,
        identifier=identifier,
        debug=args.debug,
        debug_batchnum=args.debug_batchnum,
        lr=args.lr,
        lr_warm_up_num=args.lr_warm_up_num,
        grad_clip=args.grad_clip,
        decay=args.ema_decay,
        visualizer=vis,
        logger=log,
        use_scheduler=args.use_scheduler,
        use_grad_clip=args.use_grad_clip,
        use_ema=args.use_ema,
        ema=ema,
        use_early_stop=args.use_early_stop,
        early_stop=args.early_stop
    )

    # start training!
    start = datetime.now()
    trainer.train()
    print("Time of training model ", datetime.now() - start)


if __name__ == '__main__':
    main(parser.parse_args())
