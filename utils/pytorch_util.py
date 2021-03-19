"""
This file belongs to the MultiBodySync code repository and is distributed for free.
Author: Jiahui Huang <huang-jh18@mails.tsinghua.edu.cn>
"""


import tqdm
import torch
import torch.nn as nn
import numpy as np
import shutil
from pathlib import Path
from tensorboardX import SummaryWriter
from collections import OrderedDict
import warnings


class RunningAverageMeter:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.loss_dict = OrderedDict()

    def append_loss(self, losses):
        for loss_name, loss_val in losses.items():
            if loss_val is None:
                continue
            loss_val = float(loss_val)
            if np.isnan(loss_val):
                continue
            if loss_name not in self.loss_dict.keys():
                self.loss_dict.update({loss_name: loss_val})
            else:
                old_mean = self.loss_dict[loss_name]
                self.loss_dict[loss_name] = self.alpha * old_mean + (1 - self.alpha) * loss_val

    def get_loss_dict(self):
        return {k: v for k, v in self.loss_dict.items()}


class AverageMeter:
    def __init__(self):
        self.loss_dict = OrderedDict()

    def append_loss(self, losses):
        for loss_name, loss_val in losses.items():
            if loss_val is None:
                continue
            loss_val = float(loss_val)
            if np.isnan(loss_val):
                continue
            if loss_name not in self.loss_dict.keys():
                self.loss_dict.update({loss_name: [loss_val, 1]})
            else:
                self.loss_dict[loss_name][0] += loss_val
                self.loss_dict[loss_name][1] += 1

    def get_mean_loss(self):
        all_loss_val = 0.0
        all_loss_count = 0
        for loss_name, (loss_val, loss_count) in self.loss_dict.items():
            all_loss_val += loss_val
            all_loss_count += loss_count
        return all_loss_val / (all_loss_count / len(self.loss_dict))

    def get_mean_loss_dict(self):
        loss_dict = {}
        for loss_name, (loss_val, loss_count) in self.loss_dict.items():
            loss_dict[loss_name] = loss_val / loss_count
        return loss_dict

    def get_printable(self):
        text = ""
        all_loss_sum = 0.0
        for loss_name, (loss_val, loss_count) in self.loss_dict.items():
            all_loss_sum += loss_val / loss_count
            text += "(%s:%.4f) " % (loss_name, loss_val / loss_count)
        text += " sum = %.4f" % all_loss_sum
        return text


class TensorboardViz:
    def __init__(self, logdir):
        self.logdir = logdir
        self.writter = SummaryWriter(self.logdir)

    def update(self, mode, it, eval_dict):
        self.writter.add_scalars(mode, eval_dict, global_step=it)

    def flush(self):
        self.writter.flush()


def checkpoint_state(model):
    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    return {'model_state': model_state}


def save_checkpoint(state,
                    is_best,
                    filename='checkpoint',
                    bestname='model_best'):
    filename = '{}.pth.tar'.format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}.pth.tar'.format(bestname))


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                          nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                          nn.GroupNorm)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):
    def __init__(self,
                 model,
                 bn_lambda,
                 last_epoch=-1,
                 setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(
                type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_momentum(self):
        return self.lmbd(self.last_epoch)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class Trainer(object):
    def __init__(self, model, model_fn, optimizer, exp_base, lr_scheduler=None, bnm_scheduler=None):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler = (
            model, model_fn, optimizer, lr_scheduler, bnm_scheduler)
        self.exp_base = Path(exp_base)
        self.exp_base.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name, self.best_name = "current", "best"
        self.cur_epoch = 0
        self.training_best, self.eval_best = {}, {}
        self.viz = TensorboardViz(logdir=str(self.exp_base / 'tensorboard'))

    def _train_it(self, it, batch):
        self.model.train()

        if self.lr_scheduler is not None:
            warnings.filterwarnings("ignore", category=UserWarning)
            self.lr_scheduler.step(it)
            warnings.filterwarnings("default", category=UserWarning)
        if self.bnm_scheduler is not None:
            self.bnm_scheduler.step(it)

        self.optimizer.zero_grad()
        loss, eval_res = self.model_fn(self.model, batch)  # Forwarding

        try:
            loss.backward()
        except RuntimeError:
            # This can happen if SVD does not converge or output nan values.
            return eval_res

        for param in self.model.parameters():
            if param.grad is not None and torch.any(torch.isnan(param.grad)):
                return eval_res

        self.optimizer.step()
        return eval_res

    def eval_epoch(self, d_loader):
        if self.model is not None:
            self.model.eval()

        eval_meter = AverageMeter()
        total_loss = 0.0
        count = 1.0

        with tqdm.tqdm(enumerate(d_loader, 0), total=len(d_loader), leave=False, desc='val') as tbar:
            for i, data in tbar:
                loss, eval_res = self.model_fn(self.model, data, is_eval=True)
                total_loss += loss.item()
                count += 1
                eval_meter.append_loss(eval_res)
                tbar.set_postfix(eval_meter.get_mean_loss_dict())

        return total_loss / count, eval_meter.get_mean_loss_dict()

    def train(self, n_epochs, train_loader, test_loader=None):
        it = 0
        best_loss = 1e10

        # Save init model.
        save_checkpoint(
            checkpoint_state(self.model), True,
            filename=str(self.exp_base / self.checkpoint_name),
            bestname=str(self.exp_base / self.best_name))

        with tqdm.trange(1, n_epochs + 1, desc='epochs') as tbar, \
                tqdm.tqdm(total=len(train_loader), leave=False, desc='train') as pbar:

            for epoch in tbar:
                train_meter = AverageMeter()
                train_running_meter = RunningAverageMeter(alpha=0.3)
                self.cur_epoch = epoch
                for batch in train_loader:
                    res = self._train_it(it, batch)
                    it += 1
                    pbar.update()
                    train_running_meter.append_loss(res)
                    pbar.set_postfix(train_running_meter.get_loss_dict())

                    tbar.refresh()
                    for loss_name, loss_val in res.items():
                        self.viz.update('train/' + loss_name, it, {'scalar': loss_val})
                    train_meter.append_loss(res)

                    if (it % len(train_loader)) == 0:
                        pbar.close()

                        if test_loader is not None:
                            val_loss, res = self.eval_epoch(test_loader)
                            train_avg = train_meter.get_mean_loss_dict()
                            for meter_key, meter_val in train_avg.items():
                                self.viz.update("epoch_sum/" + meter_key, it, {'train': meter_val,
                                                                               'val': np.mean(res[meter_key])})

                            is_best = val_loss < best_loss
                            best_loss = min(best_loss, val_loss)
                            save_checkpoint(
                                checkpoint_state(self.model),
                                is_best,
                                filename=str(self.exp_base / self.checkpoint_name),
                                bestname=str(self.exp_base / self.best_name))

                        pbar = tqdm.tqdm(
                            total=len(train_loader), leave=False, desc='train')
                        pbar.set_postfix(dict(total_it=it))

                    self.viz.flush()

        return best_loss
