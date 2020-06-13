# come from https://github.com/utkuozbulak/pytorch-cnn-visualizations
"""
Created on Thu Oct 26 11:06:51 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from utils.cam_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        # x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x

    def forward_pass_my(self, x):
        if "VGG" in self.model._get_name():
            conv_output, model_output = self.forward_pass(x)
            return conv_output, model_output
        elif "ShuffleNetV1" in self.model._get_name():
            conv_output = None
            x = self.model.first_conv(x)
            x = self.model.maxpool(x)
            for module_pos, module in self.model.features._modules.items():
                # x = module(x)  # Forward
                if int(module_pos) == self.target_layer:
                    x_proj = x
                    x = module.branch_main_1(x)
                    # x = module.semodule(x)
                    # x = module.channel_shuffle(x)
                    x = module.fuse(x)
                    x = module.branch_main_2(x)
                    x.register_hook(self.save_gradient)
                    conv_output = x
                    x = F.relu(x + x_proj)
                else:
                    x = module(x)

            x = self.model.globalpool(x)
            x = x.contiguous().view(-1, self.model.stage_out_channels[-1])
            x = self.model.classifier(x)
            return conv_output, x
        elif "ShuffleNetV2" in self.model._get_name():
            conv_output = None
            x = self.model.conv1(x)
            x = self.model.maxpool(x)
            x = self.model.stage2(x)
            x = self.model.stage3(x)
            x = self.model.stage4(x)
            x = self.model.conv5(x)
            x.register_hook(self.save_gradient)
            conv_output = x
            x = x.mean([2, 3])  # globalpool
            x = self.model.fc(x)

            return conv_output, x
        elif "ResNet" in self.model._get_name():
            conv_output = None
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            for module_pos, module in self.model.layer4._modules.items():
                # x = module(x)  # Forward
                if int(module_pos) == self.target_layer:
                    x = module(x)
                    x.register_hook(self.save_gradient)
                    conv_output = x
                else:
                    x = module(x)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)
            return conv_output, x

class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        batch_size = input_image.shape[0]
        conv_output, model_output = self.extractor.forward_pass_my(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(batch_size, model_output.size()[-1]).zero_()
        one_hot_output[:,target_class] = 1
        one_hot_output = one_hot_output.to(input_image.device)
        # Zero grads
        if  self.model._get_name() in ["ShuffleNetV2", "ResNet"]:
            self.model.zero_grad()
        else:
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.cpu().data.numpy()[0]
        # Get convolution outputs
        target = conv_output.cpu().data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[3],
                       input_image.shape[2]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam


if __name__ == '__main__':
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')