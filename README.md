# AneNet
The source code of AneNet, where the paper now is being reviewed. 
AneNet is proposed to screen animia based on the retinal vessel optical 
coherence tomography (OCT) images. Experimental results show that the 
proposed method achieves the state-of-the-art performance (98.65±0.7%, 
99.83±0.1%, 98.38±0.9%, 95.94±0.3% on the accuracy, AUC, sensitivity, specificity)
in our dataset.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

## Abstracti
Coming soon...

## Requirements (Other version may be ok)
* Python >= 3.5.6
* Scikit-learn == 0.21.3
* PyTorch == 1.2.0
* tqdm
* Opencv-python
* Albumentations
* H5py
* tensorboardX == 1.9.0


## Folder Structure
  ```
  pytorch-template/
  │
  ├── train_cla.py - main script to start training
  ├── test_cla.py - evaluation of trained model
  ├── configs/ - abstract base classes
  │   └── config_XXX.json - holds configuration for training
  ├
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── experiments_saved/
  │   ├── models/ - trained models are saved here
  │   
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

## Usage
The code in this repo is an MNIST example of the template.
Try `python train_cla.py --config configs/AneNet_exp_15_6.json` to run code.

Attention!!

If you don't have the dataset, please run `python VessellOCT.py` in the folder `data_loader` to generate the random dataset. When running it, please assign the correct `data_root` variable in the file `VesselOCT.py`.

`data_root` in `config_XXX.json` should also be changed to keep same.

For the cam visulization, please run `python cams_vis.py --config configs/AneNet_exp_15_6.json`

## Acknowledgements
This project is modified by the project [Pytorch-Project-Template](https://github.com/moemen95/Pytorch-Project-Template)
