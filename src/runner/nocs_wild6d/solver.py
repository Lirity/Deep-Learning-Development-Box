import os
import time

import torch
import gorilla

from tensorboardX import SummaryWriter

class Writer():
    def __init__(self, dir_project, num_counter, get_sum):
        if not os.path.isdir(dir_project):
            os.makedirs(dir_project)
        if get_sum:
            writer = SummaryWriter(dir_project)
        else:
            writer = None
        # writer = SummaryWriter(dir_project)
        self.writer = writer
        self.num_counter = num_counter
        self.list_couter = []
        for i in range(num_counter):
            self.list_couter.append(0)

    def update_scalar(self, list_name, list_value, index_counter, prefix):
        for name, value in zip(list_name, list_value):
            self.writer.add_scalar(prefix+name, float(value), self.list_couter[index_counter])

        self.list_couter[index_counter] += 1

    def refresh(self):
        for i in range(self.num_counter):
            self.list_couter[i] = 0

class Solver(gorilla.solver.BaseSolver):
    def __init__(self, model, loss, dataloaders, logger, cfg):
        super(Solver, self).__init__(model=model, dataloaders=dataloaders, cfg=cfg, logger=logger)

        self.loss = loss        # loss加入类属性提供给类内全局使用
        self.logger = logger    # logger加入类属性提供给类内全局使用
        
        tb_writer = Writer(
            dir_project=cfg.log_dir, num_counter=2, get_sum=False)
        tb_writer.writer = self.tb_writer
        self.tb_writer = tb_writer

        self.per_write = cfg.per_write

        if cfg.checkpoint_epoch != -1:
            logger.info("=> loading checkpoint from epoch {} ...".format(cfg.checkpoint_epoch))
            checkpoint = os.path.join(cfg.log_dir, 'epoch_' + str(cfg.checkpoint_epoch) + '.pth')
            
            checkpoint_file = gorilla.solver.resume(model=model, filename=checkpoint, optimizer=self.optimizer, scheduler=self.lr_scheduler)
            start_epoch = checkpoint_file['epoch'] + 1
            start_iter = checkpoint_file['iter']
            del checkpoint_file
        else:
            start_epoch = 1
            start_iter = 0

        self.epoch = start_epoch
        self.iter = start_iter


    def solve(self):
        while self.epoch <= self.cfg.max_epoch:
            self.logger.info('\nEpoch {} :'.format(self.epoch))
            end = time.time()
            dict_info_train = self.train()
            train_time = time.time()-end

            dict_info = {'train_time(min)': train_time/60.0}
            for key, value in dict_info_train.items():
                if 'loss' in key:
                    dict_info['train_'+key] = value
            
            epoch = self.epoch
            ckpt_path = os.path.join(
                self.cfg.log_dir, f'epoch_{epoch}.pth')
            gorilla.solver.save_checkpoint(
                model=self.model, filename=ckpt_path, optimizer=self.optimizer, scheduler=self.lr_scheduler, meta={'iter': self.iter, "epoch": self.epoch})

            prefix = 'Epoch {} - '.format(self.epoch)
            write_info = self.get_logger_info(prefix, dict_info=dict_info)
            self.logger.warning(write_info)
            self.epoch += 1
            

    def train(self):
        mode = 'train'
        self.model.train()
        end = time.time()
        
        for i, data in enumerate(self.dataloaders["train"]):
            data_time = time.time()-end
            
            self.optimizer.zero_grad()
            loss, dict_info_step = self.step(data, mode)
            forward_time = time.time()-end-data_time

            loss.backward()
            self.optimizer.step()
            backward_time = time.time() - end - forward_time-data_time

            dict_info_step.update({
                'T_data': data_time,
                'T_forward': forward_time,
                'T_backward': backward_time,
            })
            self.log_buffer.update(dict_info_step)

            if i % self.per_write == 0:
                # ipdb.set_trace()
                self.log_buffer.average(self.per_write)
                prefix = '[{}/{}][{}/{}][{}] Train - '.format(
                    self.epoch, self.cfg.max_epoch, i, len(self.dataloaders["train"]), self.iter)
                write_info = self.get_logger_info(
                    prefix, dict_info=self.log_buffer._output)
                self.logger.info(write_info)
                self.write_summary(self.log_buffer._output, mode)
            
            end = time.time()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.iter += 1


        dict_info_epoch = self.log_buffer.avg
        self.log_buffer.clear()

        return dict_info_epoch
    
    def step(self, data, mode):
        torch.cuda.synchronize()
        for key in data:
            data[key] = data[key].cuda()
        end_points = self.model(data)
        dict_info = self.loss(end_points)
        loss_all = dict_info['loss']

        for key in dict_info:
            dict_info[key] = float(dict_info[key].item())

        if mode == 'train':
            dict_info['lr'] = self.lr_scheduler.get_last_lr()[0]

        return loss_all, dict_info

    def get_logger_info(self, prefix, dict_info):
        info = prefix
        for key, value in dict_info.items():
            if 'T_' in key:
                info = info + '{}: {:.3f}\t'.format(key, value)
            elif 'lr' in key:
                info = info + '{}: {:.6f}\t'.format(key, value)
            else:
                info = info + '{}: {:.5f}\t'.format(key, value)

        return info
    
    def write_summary(self, dict_info, mode):
        keys = list(dict_info.keys())
        values = list(dict_info.values())
        if mode == "train":
            self.tb_writer.update_scalar(
                list_name=keys, list_value=values, index_counter=0, prefix="train_")
        elif mode == "eval":
            self.tb_writer.update_scalar(
                list_name=keys, list_value=values, index_counter=1, prefix="eval_")
        else:
            assert False