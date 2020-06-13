import os
import argparse
import pandas as pd
import collections
import torch
import numpy as np
import data_loader.VesselOCT as module_data
# from torchvision.utils import make_grid
# import torch.nn as nn
import losses.our_loss as module_loss
import model.metric as module_metric
# import model.vgg as module_arch
import model.used_models as module_arch
# import model.resnet as module_arch
# import model.unet as module_arch
from parse_config import ConfigParser
from utils.util import compute_params
from trainer import Trainer
from logger.visualization import plot_performances

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    print(config)
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    params = compute_params(model)
    print('The params is: ', params)

    logger.info(model)
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    # criterion = nn.BCEWithLogitsLoss()
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader.train_loader,
                      valid_data_loader=data_loader.valid_loader,
                      lr_scheduler=lr_scheduler)

    # Full training logic
    not_improved_count = 0
    keys = []
    values = []
    try:
        for epoch in range(trainer.start_epoch, trainer.epochs + 1):
            result = _train_epoch(trainer, epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # check the numbers of columns in resume_csv file
            if config['trainer']['resume']:
                data = pd.read_csv(trainer.csv_path)
                assert len(log) == len(data.keys()), 'the columns of resume csv file should ' \
                                                     'be same as the number of current length of log'
                del data

            # print logged informations to the screen
            local_values = []
            for key in sorted(log.keys()):
                value = log[key]
            # for key, value in log.items():
                trainer.logger.info('    {:15s}: {}'.format(str(key), round(value, 4)))
                if epoch == trainer.start_epoch:
                    keys.append(key)
                local_values.append(round(value, 4))
            # write the metrics into the csv file
            values.append(local_values.copy())
            local_values.clear()

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if trainer.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (trainer.mnt_mode == 'min' and log[trainer.mnt_metric] <= trainer.mnt_best) or \
                               (trainer.mnt_mode == 'max' and log[trainer.mnt_metric] >= trainer.mnt_best)
                except KeyError:
                    trainer.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(trainer.mnt_metric))
                    trainer.mnt_mode = 'off'
                    improved = False

                if improved:
                    trainer.mnt_best = log[trainer.mnt_metric]
                    not_improved_count = 0
                    best = True
                    trainer._save_checkpoint(epoch, save_best=best)
                else:
                    not_improved_count += 1

                if not_improved_count > trainer.early_stop:
                    trainer.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(trainer.early_stop))
                    break

            if epoch % trainer.save_period == 0:
                trainer._save_checkpoint(epoch, save_best=best)
        # the end for each epoch
        if not config['trainer']['resume']:
            trainer.csv_writer.writerow(keys)
        trainer.csv_writer.writerows(values)
        trainer.f.close()
        plot_performances(trainer.csv_path)

    except KeyboardInterrupt:
        if not config['trainer']['resume']:
            trainer.csv_writer.writerow(keys)
        trainer.csv_writer.writerows(values)
        trainer.f.close()
        plot_performances(trainer.csv_path)
        print('The Crtl + C is pressed!')
def _train_epoch(trainer, epoch):
    """
    Training logic for an epoch

    :param epoch: Integer, current training epoch.
    :return: A log that contains average loss and metric in this epoch.
    """
    trainer.model.train()
    trainer.train_metrics.reset()
    outputs = []# [[batch, 1, 1,1]]
    targets = []
    for batch_idx, (data, target) in enumerate(trainer.data_loader):
        data, target = data.to(trainer.device), target.to(trainer.device).float()
        # assert not torch.isnan(data), 'the input data is None with batch_idx: {}'.format(batch_idx)
        # assert not torch.isinf(data), 'the input data is Inf with batch_idx: {}'.format(batch_idx)

        trainer.optimizer.zero_grad()
        output = trainer.model(data)
        output = output.unsqueeze(dim=2).unsqueeze(dim=3)
        target = target.unsqueeze(dim=2)
        loss = trainer.criterion(output, target)
        if np.isnan(float(loss.item())):
            raise ValueError('Loss is nan during training...')
        if np.isinf(float(loss.item())):
            raise ValueError('Loss is Inf during training...')
        loss.backward()
        trainer.optimizer.step()

        trainer.writer.set_step((epoch - 1) * trainer.len_epoch + batch_idx)
        trainer.train_metrics.update('loss', loss.item())
        outputs.append(output.detach().clone())
        targets.append(target.detach().clone())
        # for met in trainer.metric_ftns:
        #     trainer.train_metrics.update(met.__name__, met(output, target))

        if batch_idx % trainer.log_step == 0:
            trainer.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                epoch,
                trainer._progress(batch_idx),
                loss.item()))
            # trainer.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # if batch_idx == trainer.len_epoch:
        #     break
    outputs = torch.cat(outputs, dim=0) # [steps*batch, 1, 1, 1]
    targets = torch.cat(targets, dim=0)
    for met in trainer.metric_ftns:
        trainer.train_metrics.update(met.__name__, met(outputs, targets))
    log = trainer.train_metrics.result()

    if trainer.do_validation:
        val_log = _valid_epoch(trainer, epoch)
        log.update(**{'val_'+k : v for k, v in val_log.items()})

    if trainer.lr_scheduler is not None:
        trainer.lr_scheduler.step()
    return log

def _valid_epoch(trainer, epoch):
    """
    Validate after training an epoch

    :param epoch: Integer, current training epoch.
    :return: A log that contains information about validation
    """
    trainer.model.eval()
    trainer.valid_metrics.reset()
    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(trainer.valid_data_loader):
            data, target = data.to(trainer.device), target.to(trainer.device).float()

            output = trainer.model(data)
            output = output.unsqueeze(dim=2).unsqueeze(dim=3)
            target = target.unsqueeze(dim=2)
            loss = trainer.criterion(output, target)
            if np.isnan(float(loss.item())):
                raise ValueError('Loss is nan during training...')
            if np.isinf(float(loss.item())):
                raise ValueError('Loss is Inf during training...')

            outputs.append(output.detach().clone())
            targets.append(target.detach().clone())
            trainer.writer.set_step((epoch - 1) * len(trainer.valid_data_loader) + batch_idx, 'valid')
            trainer.valid_metrics.update('loss', loss.item())
            # for met in trainer.metric_ftns:
            #     trainer.valid_metrics.update(met.__name__, met(output, target))
            # trainer.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

    # add histogram of model parameters to the tensorboard
    # for name, p in trainer.model.named_parameters():
    #     trainer.writer.add_histogram(name, p, bins='auto')
    outputs = torch.cat(outputs, dim=0) # [steps*batch, 1, 1, 1]
    targets = torch.cat(targets, dim=0)
    for met in trainer.metric_ftns:
        trainer.valid_metrics.update(met.__name__, met(outputs, targets))
    return trainer.valid_metrics.result()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
