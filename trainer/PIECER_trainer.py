# -*- coding: utf-8 -*-
"""
Trainer file for SQuAD dataset.
"""
import os
import shutil
import time
import torch
import torch.nn.functional as F
from datetime import datetime
from .metric import convert_tokens, evaluate_by_dict
from util.file_utils import pickle_load_large_file


class Trainer(object):

    def __init__(self, args, model, loss,
                 train_data_loader, dev_data_loader,
                 train_eval_file, dev_eval_file,
                 optimizer, scheduler, epochs, with_cuda,
                 save_dir, verbosity=2, save_freq=1, print_freq=10,
                 resume=False, identifier='',
                 debug=False, debug_batchnum=2,
                 visualizer=None, logger=None,
                 grad_clip=5.0, decay=0.9999,
                 lr=0.001, lr_warm_up_num=1000,
                 use_scheduler=False, use_grad_clip=False,
                 use_ema=False, ema=None,
                 use_early_stop=False, early_stop=10):
        self.device = torch.device("cuda" if with_cuda else "cpu")
        self.args = args

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.print_freq = print_freq
        self.verbosity = verbosity
        self.identifier = identifier
        self.visualizer = visualizer
        self.with_cuda = with_cuda

        self.train_data_loader = train_data_loader
        self.train_eval_dict = pickle_load_large_file(train_eval_file)
        self.dev_data_loader = dev_data_loader
        self.dev_eval_dict = pickle_load_large_file(dev_eval_file)
        self.is_debug = debug
        self.debug_batchnum = debug_batchnum
        self.logger = logger
        self.unused = True  # whether scheduler has been updated

        self.lr = lr
        self.lr_warm_up_num = lr_warm_up_num
        self.decay = decay
        self.use_scheduler = use_scheduler
        self.scheduler = scheduler
        self.use_grad_clip = use_grad_clip
        self.grad_clip = grad_clip
        self.use_ema = use_ema
        self.ema = ema
        self.use_early_stop = use_early_stop
        self.early_stop = early_stop

        self.start_time = datetime.now().strftime('%b-%d_%H-%M')
        self.start_epoch = 1
        self.step = 0
        self.best_em = 0
        self.best_f1 = 0
        if resume:
            self._resume_checkpoint(resume)
            self.model = self.model.to(self.device)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        if self.use_ema:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.ema.register(name, param.data)

    def train(self):
        patience = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            # if self.args.use_kg_gcn:
            #     self.model.save_tools()

            if self.use_early_stop:
                if result["f1"] < self.best_f1 and result["em"] < self.best_em:
                    patience += 1
                    if patience > self.early_stop:
                        print("Perform early stop!")
                        break
                else:
                    patience = 0

            is_best = False
            if result["f1"] > self.best_f1 or result["f1"] == self.best_f1 and result["em"] > self.best_em:
                is_best = True
                self.best_f1 = result["f1"]
                self.best_em = result["em"]

            if epoch % self.save_freq == 0:
                self._save_checkpoint(epoch, result["f1"], result["em"], is_best)

    def _train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.device)

        # initialize
        global_loss = 0.0
        last_step = self.step - 1
        last_time = time.time()

        # !!!!!!! 防止小 bug, 最好在所有更新之前清零一下梯度
        self.optimizer.zero_grad()

        # train over batches
        for batch_idx, batch in enumerate(self.train_data_loader):
            # get batch
            (context_wids,
            context_cids,
            question_wids,
            question_cids,
            y1,
            y2,
            y1s,
            y2s,
            id,
            context_tokens, 
            ques_tokens,
            context_ner_ids
            ) = batch
            batch_num, question_len = question_wids.size()
            _, context_len = context_wids.size()
            context_wids = context_wids.to(self.device)
            context_cids = context_cids.to(self.device)
            question_wids = question_wids.to(self.device)
            question_cids = question_cids.to(self.device)
            context_ner_ids = context_ner_ids.to(self.device)
            y1 = y1.to(self.device)
            y2 = y2.to(self.device)
            id = id.to(self.device)

            # calculate loss
            p1, p2 = self.model(
                context_wids,
                context_cids,
                question_wids,
                question_cids, 
                context_tokens, 
                ques_tokens, 
                context_ner_ids
            )

            loss1 = self.loss(p1, y1)
            loss2 = self.loss(p2, y2)
            loss = torch.mean(loss1 + loss2)

            if self.args.acc_batch == 1:
                self.optimizer.zero_grad()
                loss.backward()
                global_loss += loss.item()

                # gradient clip
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                # update model
                self.optimizer.step()

                # update learning rate
                if self.use_scheduler:
                    self.scheduler.step()

                # exponential moving avarage
                if self.use_ema and self.ema is not None:
                    self.ema(self.model, self.step)
            else: 
                # !!!!!!! 累计梯度的 batch_size
                # 模拟 batch_size * acc_batch 的平均 loss
                loss = loss / self.args.acc_batch
                loss.backward()
                global_loss += loss.item()

                if batch_idx % self.args.acc_batch == 0 or batch_idx == len(self.train_data_loader) - 1: 
                    # gradient clip
                    if self.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    # update model
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # exponential moving avarage
                    if self.use_ema and self.ema is not None:
                        self.ema(self.model, self.step // 4)
                    # update learning rate
                    if self.use_scheduler:
                        self.scheduler.step()

            # print training info
            if self.step % self.print_freq == self.print_freq - 1:
                used_time = time.time() - last_time
                step_num = self.step - last_step
                speed = self.train_data_loader.batch_size * step_num / used_time
                batch_loss = global_loss / step_num * self.args.acc_batch
                print(("step: {}/{} \t "
                    "epoch: {} \t "
                    "lr: {} \t "
                    "loss: {} \t "
                    "speed: {} examples/sec").format(
                        batch_idx, len(self.train_data_loader),
                        epoch,
                        self.scheduler.get_last_lr(),
                        batch_loss,
                        speed))
                global_loss = 0.0
                last_step = self.step
                last_time = time.time()
            self.step += 1

            if self.is_debug and batch_idx >= self.debug_batchnum:
                break

        train_metrics = self._valid_eopch(self.train_eval_dict, self.train_data_loader, valid_data='training')
        print("train_em: %f \t train_f1: %f" % (train_metrics["exact_match"], train_metrics["f1"]))

        metrics = self._valid_eopch2(self.dev_eval_dict, self.dev_data_loader, valid_data='dev')
        print("dev_em: %f \t dev_f1: %f" % (metrics["exact_match"], metrics["f1"]))

        result = {}
        result["em"] = metrics["exact_match"]
        result["f1"] = metrics["f1"]
        return result

    def _valid_eopch(self, eval_dict, data_loader, valid_data='dev'):
        """
        Evaluate model over development dataset.
        Return the metrics: em, f1.
        """
        if self.use_ema and self.ema is not None:
            self.ema.assign(self.model)

        self.model.eval()
        answer_dict = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                (context_wids,
                context_cids,
                question_wids,
                question_cids,
                y1,
                y2,
                y1s,
                y2s,
                id,
                context_tokens, 
                ques_tokens,
                context_ner_ids
                ) = batch
                context_wids = context_wids.to(self.device)
                context_cids = context_cids.to(self.device)
                question_wids = question_wids.to(self.device)
                question_cids = question_cids.to(self.device)
                context_ner_ids = context_ner_ids.to(self.device)
                y1 = y1.to(self.device)
                y2 = y2.to(self.device)

                p1, p2 = self.model(
                    context_wids,
                    context_cids,
                    question_wids,
                    question_cids, 
                    context_tokens, 
                    ques_tokens, 
                    context_ner_ids
                )

                p1 = F.softmax(p1, dim=1)
                p2 = F.softmax(p2, dim=1)
                outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
                for j in range(outer.size()[0]):
                    outer[j] = torch.triu(outer[j])
                a1, _ = torch.max(outer, dim=2)
                a2, _ = torch.max(outer, dim=1)
                ymin = torch.argmax(a1, dim=1)
                ymax = torch.argmax(a2, dim=1)
                answer_dict_, _ = convert_tokens(eval_dict, id.tolist(), ymin.tolist(), ymax.tolist())
                answer_dict.update(answer_dict_)

                # if batch_idx + 1 == self.args.val_num_batches:  # !!!!! 这里暂时不要退出，验证集上的结果全部测出来
                #     break

                if valid_data == 'training' and batch_idx + 1 == self.args.val_num_batches:  # !!!!! 对于 training 集的 evaluation，可以只测一部分以节约时间
                    break

                if self.is_debug and batch_idx >= self.debug_batchnum:
                    break

        metrics = evaluate_by_dict(eval_dict, answer_dict)
        if self.use_ema and self.ema is not None:
            self.ema.resume(self.model)
        self.model.train()
        return metrics

    def _valid_eopch2(self, eval_dict, data_loader, valid_data='dev'):
        """
        Evaluate model over development dataset.
        Return the metrics: em, f1.
        """
        if self.use_ema and self.ema is not None:
            self.ema.assign(self.model)

        self.model.eval()
        answer_dict = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                (context_wids,
                context_cids,
                question_wids,
                question_cids,
                y1,
                y2,
                y1s,
                y2s,
                id,
                context_tokens, 
                ques_tokens,
                context_ner_ids
                ) = batch
                context_wids = context_wids.to(self.device)
                context_cids = context_cids.to(self.device)
                question_wids = question_wids.to(self.device)
                question_cids = question_cids.to(self.device)
                context_ner_ids = context_ner_ids.to(self.device)
                y1 = y1.to(self.device)
                y2 = y2.to(self.device)

                p1, p2 = self.model(
                    context_wids,
                    context_cids,
                    question_wids,
                    question_cids, 
                    context_tokens, 
                    ques_tokens, 
                    context_ner_ids
                )

                p1 = F.softmax(p1, dim=1)  # batch, len
                for i in range(p1.shape[0]): 
                    for j in range(p1.shape[1]): 
                        if context_ner_ids[i][j] != 1 and context_ner_ids[i][j] != 4:
                            p1[i][j] = 0.0
                max_p1, ymin = torch.max(p1, dim=1)  # batch
                ymax = torch.zeros_like(ymin)
                for i in range(ymin.shape[0]): 
                    now = ymin[i].item()
                    while True: 
                        if context_ner_ids[i][now] == 3 or context_ner_ids[i][now] == 4: 
                            break
                        if now == context_ner_ids.shape[1] - 1:
                            break
                        if context_ner_ids[i][now + 1] == 0: 
                            break
                        now += 1
                    ymax[i] = now

                # p1 = F.softmax(p1, dim=1)
                # p2 = F.softmax(p2, dim=1)
                # outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
                # for j in range(outer.size()[0]):
                #     outer[j] = torch.triu(outer[j])
                # a1, _ = torch.max(outer, dim=2)
                # a2, _ = torch.max(outer, dim=1)
                # ymin = torch.argmax(a1, dim=1)
                # ymax = torch.argmax(a2, dim=1)
                answer_dict_, _ = convert_tokens(eval_dict, id.tolist(), ymin.tolist(), ymax.tolist())
                answer_dict.update(answer_dict_)

                # if batch_idx + 1 == self.args.val_num_batches:  # !!!!! 这里暂时不要退出，验证集上的结果全部测出来
                #     break

                if valid_data == 'training' and batch_idx + 1 == self.args.val_num_batches:  # !!!!! 对于 training 集的 evaluation，可以只测一部分以节约时间
                    break

                if self.is_debug and batch_idx >= self.debug_batchnum:
                    break

        metrics = evaluate_by_dict(eval_dict, answer_dict)
        if self.use_ema and self.ema is not None:
            self.ema.resume(self.model)
        self.model.train()
        return metrics

    def _save_checkpoint(self, epoch, f1, em, is_best):
        if self.use_ema and self.ema is not None:
            self.ema.assign(self.model)
        arch = type(self.model).__name__
        state = {
            'epoch': epoch,
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'best_em': self.best_em,
            'step': self.step + 1,
            'start_time': self.start_time
        }
        filename = os.path.join(
            self.save_dir,
            self.identifier + 'checkpoint_epoch{:02d}_f1_{:.5f}_em_{:.5f}.pth.tar'.format(epoch, f1, em)
        )
        print("Saving checkpoint: {} ...".format(filename))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.save_dir, 'model_best.pth.tar'))
        if self.use_ema and self.ema is not None:
            self.ema.resume(self.model)
        return filename

    def _resume_checkpoint(self, resume_path):
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_f1 = checkpoint['best_f1']
        self.best_em = checkpoint['best_em']
        self.step = checkpoint['step']
        self.start_time = checkpoint['start_time']
        if self.use_scheduler:
            self.scheduler.last_epoch = checkpoint['epoch']
        print("Checkpoint '{}' (epoch {}) loaded".format(
            resume_path, self.start_epoch))
