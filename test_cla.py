import argparse
import os
import csv
import time
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
# import data_loader.data_loaders as module_data
import data_loader.VesselOCT as module_data
# import model.loss as module_loss
import model.metric as module_metric
import model.used_models as module_arch
# import model.unet as module_arch
from parse_config import ConfigParser
from utils.util import save_resutls
from utils.util import compute_params
from utils.util import MetricTracker, reset_bn_stats
from utils.time_analysis import compute_precise_time
# from torchstat import stat
from torchsummary import summary
# import h5py

def main(config):
    # logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        mode="test",
        data_root="/root/userfolder/Dataset/ImagesAnnotations_aug/",
        fold= 0,
        num_workers=4,
        batch_size=96
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    params = compute_params(model)
    print(model)
    print('the params of model is: ', params)
    # logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    loss_fn = nn.BCEWithLogitsLoss()
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    resume_path = os.path.join(config['project_root'], config['trainer']['resume_path'])
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
    # logger.info('Loading checkpoint: {} ...'.format(resume_path))
    print('Loading checkpoint: {} ...'.format(resume_path))
    state_dict = checkpoint['state_dict']
    gpus = config['gpu_device']
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    outputs = []
    targets = []
    test_metrics = MetricTracker('loss','time', *[m.__name__ for m in metric_fns], writer=None)
    image_shape = None
    f_dir, f_name = os.path.split(resume_path)
    csv_path = os.path.join(f_dir, 'prediction.csv')
    f = open(csv_path, 'w')
    csv_writer = csv.writer(f)
    keys = ['label','pred']
    values = []
    csv_writer.writerow(keys)
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader.test_loader)):
            data, target = data.to(device), target.to(device).float()
            # data, target = data.cuda(), target.cuda().float()
            image_shape = [data.shape[2], data.shape[3]]
            torch.cuda.synchronize(device)
            start = time.time()
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            output = model(data)
            torch.cuda.synchronize(device)
            end = time.time()
            # print('time:',end-start)
            pred = output.clone() # [batch, c]
            pred_list = torch.sigmoid(pred).squeeze().tolist()
            label = target.clone() # [batch]
            label_list = label.squeeze().tolist()
            _ = [values.append([label_list[index], pred_list[index]]) for index in range(len(pred_list))]

            output = output.unsqueeze(dim=2).unsqueeze(dim=3)
            target = target.unsqueeze(dim=2)
            outputs.append(output.clone())
            targets.append(target.clone())
            loss = loss_fn(output.squeeze(dim=1), target)
            total_loss += loss.item()
            test_metrics.update('time', end - start)
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target, apply_nonlin=True)
            # print(prof)
    csv_writer.writerows(values)
    f.close()
    outputs = torch.cat(outputs, dim=0) # [steps*batch, 1, 1, 1]
    targets = torch.cat(targets, dim=0)

    for met in metric_fns:
        test_metrics.update(met.__name__, met(outputs, targets))
    log = test_metrics.result()

    print(log)
    # summary(model, (1,496, 384))
    time_results = compute_precise_time(model, [496, 384],96, loss_fn, device)
    print(time_results)
    reset_bn_stats(model)
    return


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, mode='test')
    main(config)
