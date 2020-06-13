import json
import torch
import time
import os
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)

# The settings of ReLayNet
SEG_LABELS_LIST = [
    {"id": 0, "name": "Background", "rgb_values": [0, 0, 0]},
    {"id": 1, "name": "Class 1", "rgb_values": [255, 0, 0]},
    {"id": 2, "name": "Class 2", "rgb_values": [0, 255, 0]},
    {"id": 3, "name": "Class 3", "rgb_values": [145, 165, 120]},
    {"id": 4, "name": "Class 4", "rgb_values": [255, 165, 0]},
    {"id": 5, "name": "Class 5", "rgb_values": [0, 128, 128]},
    {"id": 6, "name": "Class 6", "rgb_values": [0, 90, 0]},
    {"id": 7, "name": "Class 7", "rgb_values": [64, 0, 0]},
    {"id": 8, "name": "Class 8", "rgb_values": [12, 0, 0]},
    {"id": 9, "name": "Class 9", "rgb_values": [0, 0, 0]}]

def label_img_to_rgb(label_img):
    assert label_img.max() <= 9, \
        'The class number {} > 10, which is not supported!'.format(label_img.max()+1)
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1, 2, 3, 0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)

def save_resutls(resume_path, image, label, prediction, step):
    '''
    All the input should be the numpy.ndarray
    :param image: [batch, 1, H, W]
    :param label: [batch, H, W]
    :param prediction: [batch, C, H, W]
    :return:
    '''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # from datasets.hcms import label_img_to_rgb
    import cv2

    f_dir, f_name = os.path.split(resume_path)
    current_root = os.path.join(f_dir,'prediction')
    if not os.path.exists(current_root):
        os.makedirs(current_root)
    # reverse normalize
    mean = [0.164]
    std = [0.22]
    for channel in range(image.shape[1]):
        image[:,channel] = image[:,channel] * std[channel] + mean[channel]

    image = image * 255
    prediction_shape = prediction.shape # [batch, 1, H, W]
    if prediction_shape[1] > 1:
        prediction = prediction.argmax(axis=1) # [batch, C, H, W] to [batch, H, W]
        prediction = label_img_to_rgb(prediction)
        label = label_img_to_rgb(label)
    else:
        prediction = prediction.squeeze(axis=1) # [batch, 1, H, W] to [batch, H, W]


    batch = prediction_shape[0]
    for ind in range(batch):
        image_path = os.path.join(current_root, 'step_{}_{}_image.png'.format(step,ind))
        label_path = os.path.join(current_root, 'step_{}_{}_label.png'.format(step,ind))
        pred_path = os.path.join(current_root, 'step_{}_{}_pred.png'.format(step,ind))
        cv2.imwrite(image_path, image[ind,0])
        plt.imsave(label_path, label[ind])
        # cv2.imwrite(label_path, label[ind])
        plt.imsave(pred_path, prediction[ind])
        # cv2.imwrite(pred_path, prediction[ind])
        # print('The png images are saved in {}!'.format(results_root))
    return True

def compute_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

# come from https://github.com/sacmehta/ESPNet/issues/57
def computeTime(model, device='cuda'):
    inputs = torch.randn(1, 3, 512, 1024)
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    i = 0
    time_spent = []
    while i < 100:
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1
    print('Avg execution time (ms): {:.3f}'.format(np.mean(time_spent)))

def reset_bn_stats(model):
    """Resets running BN stats."""
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()