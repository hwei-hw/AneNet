#
import argparse
import os
import time
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
# import data_loader.data_loaders as module_data
import data_loader.VesselOCT as module_data
from PIL import Image
# import model.loss as module_loss
import model.metric as module_metric
import model.used_models as module_arch
# import model.unet as module_arch
from parse_config import ConfigParser
from utils.util import save_resutls

from utils.gradcam import GradCam
from utils.cam_functions import apply_colormap_on_image, save_image

import h5py

def main(config):
    # logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        mode="test",
        data_root="/root/userfolder/Dataset/ImagesAnnotations_aug/",
        fold= 0,
        num_workers=0,
        batch_size=1
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    print(model)
    # logger.info(model)

    model_name = config['name']
    out_dir = os.path.join(config['project_root'], "experiments_saved/vis", model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])

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

    target_layer = config['target_layer']
    grad_cam = GradCam(model, target_layer=target_layer)

    image_mean = 0.164
    image_std = 0.22

    # with torch.no_grad():
    for i, (data, target) in enumerate(tqdm(data_loader.test_loader)):
        data, target = data.to(device), target.to(device).float()
        # output = model(data)
        target_class = 0
        cam = grad_cam.generate_cam(data, target_class) # cam is ndarray with [H, W]
        org_im = data.cpu().numpy()[0,0] # [h, w]
        org_im = (org_im * image_std + image_mean)*255
        org_im = Image.fromarray(np.uint8(org_im))
        heatmap, heatmap_on_image = apply_colormap_on_image(org_im, cam, 'hsv')

        # Save colored heatmap
        path_to_file = os.path.join(out_dir, '{}.png'.format(i))
        save_image(org_im, path_to_file)
        path_to_file = os.path.join(out_dir, '{}_Cam_Heatmap.png'.format(i))
        save_image(heatmap, path_to_file)
        path_to_file = os.path.join(out_dir, '{}_Cam_Heatmap.png'.format(i))
        save_image(heatmap, path_to_file)
        # Save heatmap on iamge
        path_to_file = os.path.join(out_dir, '{}_Cam_On_Image.png'.format(i))
        save_image(heatmap_on_image, path_to_file)
        # SAve grayscale heatmap
        path_to_file = os.path.join(out_dir, '{}_Cam_Grayscale.png'.format(i))
        save_image(cam, path_to_file)





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
