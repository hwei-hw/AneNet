import os
import cv2
import numpy as np
import torch
import glob
import h5py
import tqdm
from sklearn.model_selection import KFold
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as standard_transforms
from albumentations import HorizontalFlip, ElasticTransform, Compose, Rotate, ShiftScaleRotate
import matplotlib.pyplot as plt

def get_split():
    pass

def augment_funs(input_array):
    # input_array is a 2D array
    input_fliplr = np.fliplr(input_array)
    input_flipud = np.flipud(input_array)
    input_fliplrud = np.flipud(input_fliplr)
    return (input_array, input_fliplr, input_flipud, input_fliplrud)

def augment_images(source_data_root):
    # expand the dataset by the horizon and vertical flip
    # source_data_root = '/home/qinwang/vesselOCT/ImagesAnnotations/'
    # dst_root = '/home/qinwang/vesselOCT/ImagesAnnotations_aug/'
    top_dir = source_data_root.split('/')[-2] # ImagesAnnotations
    image_files = glob.glob(os.path.join(source_data_root,'Origin')+'/*/*.bmp')
    print('the dir {} has {} .bmp files. '.format(source_data_root,len(image_files)))
    for file in tqdm.tqdm(image_files):
        image = cv2.imread(file, 0)
        file_path, file_name = os.path.split(file) # e.g. file_name = 1.bmp
        RPE_label = cv2.imread(file.replace('Origin','RPE'),0)
        Vessel_label = cv2.imread(file.replace('Origin','Vessel'),0)

        image_augs = augment_funs(image)
        RPE_label_augs = augment_funs(RPE_label)
        Vessel_label_augs = augment_funs(Vessel_label)

        file_name_num = file_name.split('.')[0]
        file_name_replaces = ['_org','_lr','_ud','_lrud']
        for index in range(len(image_augs)):
            dst_file = file.replace(top_dir, top_dir + '_aug')
            file_name_new = file_name.replace(file_name_num,file_name_num+file_name_replaces[index])
            dst_file = dst_file.replace(file_name, file_name_new)
            dst_file = dst_file.replace('bmp','png')
            if not os.path.exists(os.path.split(dst_file)[0]):
                os.makedirs(os.path.split(dst_file)[0])
            cv2.imwrite(dst_file,image_augs[index])
            dst_file = dst_file.replace('Origin','RPE')
            if not os.path.exists(os.path.split(dst_file)[0]):
                os.makedirs(os.path.split(dst_file)[0])
            cv2.imwrite(dst_file,RPE_label_augs[index])
            dst_file = dst_file.replace('RPE','Vessel')
            if not os.path.exists(os.path.split(dst_file)[0]):
                os.makedirs(os.path.split(dst_file)[0])
            cv2.imwrite(dst_file,Vessel_label_augs[index])

def make_dataset(data_root, n_splits=5):
    # generate the .npy format dataset
    image_files = glob.glob(os.path.join(data_root,'Origin')+'/*/*.png')
    print('the dir {} has {} .bmp files. '.format(data_root,len(image_files)))
    image_files.sort()
    images = []
    RPE_labels = []
    Vessel_labels = []
    image_labels = []
    for file in tqdm.tqdm(image_files):
        file_dir, file_name = os.path.split(file)
        if 'A' in file_dir.split('/')[-1]:
            image_labels.append(1)
        else:
            image_labels.append(0)

        image = cv2.imread(file,0)
        RPE_label = cv2.imread(file.replace('Origin','RPE'), 0)
        Vessel_label = cv2.imread(file.replace('Origin','Vessel'), 0)

        image = cv2.resize(image,(image.shape[1]+2,image.shape[0]), interpolation=cv2.INTER_LINEAR)
        RPE_label = cv2.resize(RPE_label,(RPE_label.shape[1]+2,RPE_label.shape[0]), interpolation=cv2.INTER_NEAREST)
        Vessel_label = cv2.resize(Vessel_label,(Vessel_label.shape[1]+2,Vessel_label.shape[0]), interpolation=cv2.INTER_NEAREST)

        images.append(image)
        RPE_labels.append(RPE_label)
        Vessel_labels.append(Vessel_label)

    images_numpy = np.array(images)
    RPE_labels_numpy = np.array(RPE_labels)
    Vessel_labels_numpy = np.array(Vessel_labels)
    image_labels_numpy = np.array(image_labels)

    # save the npy which contains all the files
    np.save(os.path.join(data_root, 'all_images.npy'), images_numpy)
    np.save(os.path.join(data_root, 'all_RPE_labels.npy'), RPE_labels_numpy)
    np.save(os.path.join(data_root, 'all_Vessel_labels.npy'), Vessel_labels_numpy)
    np.save(os.path.join(data_root, 'all_image_labels.npy'), image_labels_numpy)

    dataset_mean = (images_numpy/255.0).mean() # 0.164
    dataset_std = (images_numpy/255.0).std() # 0.222
    print('the dataset mean: {} and std: {}'.format(dataset_mean, dataset_std))
    kf = KFold(n_splits=n_splits, shuffle=True)
    kf_generator = kf.split(images_numpy)
    fold_number = 0
    for train_indexs, test_indexs in kf_generator:
        # train_indexs, test_indexs = next(kf_generator) # select the first fold
        train_set = np.array([images_numpy[train_indexs], RPE_labels_numpy[train_indexs],
                     Vessel_labels_numpy[train_indexs]])
        test_set = np.array([images_numpy[test_indexs], RPE_labels_numpy[test_indexs],
                     Vessel_labels_numpy[test_indexs]])

        # save the npy which contains all the files
        np.save(os.path.join(data_root, 'trainset_fold_{}.npy'.format(fold_number)), train_set)
        np.save(os.path.join(data_root, 'testset_fold_{}.npy'.format(fold_number)), test_set)
        fold_number = fold_number + 1

    return True

def make_random_data(data_root):
    # generate the random data in data_root
    all_images = np.random.random((1369+343, 496, 384))*255
    all_images = all_images.astype(np.uint8)
    all_labels = np.random.randint(0,2,1369+343)

    np.save(os.path.join(data_root, 'all_images.npy'), all_images)
    np.save(os.path.join(data_root, 'all_image_labels.npy'), all_labels)
    print('The random data has been generated!')

def make_dataset_cla(data_root, n_splits=5):
    # for the classification
    images_numpy = np.load(os.path.join(data_root, 'all_images.npy')) # the original images [number, 496, 384]
    image_labels_numpy = np.load(os.path.join(data_root, 'all_image_labels.npy')) # the image-level label [number]

    kf = KFold(n_splits=n_splits, shuffle=True)
    kf_generator = kf.split(images_numpy)
    fold_number = 0

    for train_indexs, test_indexs in kf_generator:
        # train_indexs, test_indexs = next(kf_generator) # select the first fold
        train_images = images_numpy[train_indexs]
        train_labels = image_labels_numpy[train_indexs]
        test_images = images_numpy[test_indexs]
        test_labels = image_labels_numpy[test_indexs]

        # save the h5 file
        h5_path = os.path.join(data_root, 'cla_data_fold_{}.h5'.format(fold_number))
        h5f = h5py.File(h5_path, 'w')
        h5f.create_dataset('train_images', data=train_images) # []
        h5f.create_dataset('train_labels', data=train_labels) # []
        h5f.create_dataset('test_images', data=test_images)
        h5f.create_dataset('test_labels', data=test_labels)
        h5f.create_dataset('train_indexs', data=train_indexs)
        h5f.create_dataset('test_indexs', data=test_indexs)
        h5f.close()
        print('The data of fold {} is saved in {}'.format(fold_number, data_root))

        fold_number = fold_number + 1

    return True


class VesselOCTCla(data.Dataset):
    def __init__(self, mode, data_root, fold = 0, joint_transform=None, transform=None, target_transform=None):

        self.mode = mode
        assert self.mode in ['train', 'val', 'test']

        self.data_root = data_root
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

        data_fold = h5py.File(os.path.join(self.data_root, 'cla_data_fold_{}.h5'.format(fold)), 'r')
        self.imgs = data_fold['{}_images'.format(self.mode)][()] # [number, h, w]
        self.labels = data_fold['{}_labels'.format(self.mode)][()] # [number,]

        # del data

    def __getitem__(self, index):
        image = self.imgs[index] # the 2D array
        label = self.labels[index] # the 1D array
        image = image[:,:,np.newaxis] # [h, w, 1]

        # augmentation
        if self.mode == 'train':
            augmentation = Compose([ShiftScaleRotate(),
                                    ElasticTransform(p=0.5, alpha=50, sigma=120 * 0.05, alpha_affine=120 * 0.03)], p=0.5)
            augmentated = augmentation(image=image)
            image = augmentated['image']

        if self.transform is not None:
            image = self.transform(image)
        image_label = torch.tensor(np.array([label])).float()

        return image, image_label

    def __len__(self):
        return self.imgs.shape[0]

class VesselOCTClaDataLoader:
    def __init__(self, mode, data_root, fold, num_workers, batch_size):
        # self.config = config
        self.mode = mode
        assert self.mode in ['train', 'test', 'val']
        self.data_root = data_root
        self.fold = fold
        self.data_loader_workers = num_workers
        # self.batch_size = batch_size
        self.train_batch_size = batch_size
        self.val_batch_size = batch_size
        self.test_batch_size = batch_size
        self.pin_memory = False

        mean_std = ([0.164], [0.222])
        self.input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        # self.target_transform = standard_transforms.Compose([
        #     standard_transforms.ToTensor(),
        # ])
        self.target_transform = None

        if self.mode == 'train':
            train_set = VesselOCTCla('train', self.data_root,self.fold,
                            transform=self.input_transform, target_transform=self.target_transform)
            valid_set = VesselOCTCla('test', self.data_root,self.fold,
                            transform=self.input_transform, target_transform=self.target_transform)

            self.train_loader = DataLoader(train_set, batch_size=self.train_batch_size, shuffle=True,
                                           num_workers=self.data_loader_workers,
                                           pin_memory=self.pin_memory)
            self.valid_loader = DataLoader(valid_set, batch_size=self.val_batch_size, shuffle=False,
                                           num_workers=self.data_loader_workers,
                                           pin_memory=self.pin_memory)
            self.train_iterations = (len(train_set) + self.train_batch_size) // self.train_batch_size
            self.valid_iterations = (len(valid_set) + self.val_batch_size) // self.val_batch_size

        elif self.mode == 'test':
            test_set = VesselOCTCla('test', self.data_root,self.fold,
                           transform=self.input_transform, target_transform=self.target_transform)

            self.test_loader = DataLoader(test_set, batch_size=self.test_batch_size, shuffle=False,
                                          num_workers=self.data_loader_workers,
                                          pin_memory=self.pin_memory)
            self.test_iterations = (len(test_set) + self.test_batch_size) // self.test_batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass


def check_data_loader_cla(data_root, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mode = 'test'
    # data_fold = h5py.File(os.path.join(data_root, 'seg_cla_data_fold_0.h5'), 'r')
    data_fold = h5py.File(os.path.join(data_root, 'cla_data_fold_0.h5'), 'r')
    # labels_h5 = data_fold['{}_image_labels'.format(mode)].value  # [number,]
    labels_h5 = data_fold['{}_labels'.format(mode)].value  # [number,]
    images_h5 = data_fold['{}_images'.format(mode)].value  # [number,]

    batch_size = 12
    data_loader = VesselOCTClaDataLoader(
        mode=mode,
        data_root=data_root,
        fold= 0,
        num_workers=8,
        batch_size=batch_size
    )
    dataiter = iter(data_loader.test_loader)

    mean = [0.164]
    std = [0.22]

    for index in range(5):
        images, labels = dataiter.next()
        # data = dataiter.next()
        # images = data['image']
        # labels = data['label']
        # labels = labels.numpy()
        print(labels.numpy())
        print(labels_h5[index*batch_size:(index+1)*batch_size])
        print('...................................')
        for i in range(batch_size):
            image = images[i,0].numpy()
            image = image * std[0] + mean[0]
            image_path = os.path.join(output_dir, 'data_loader_{}.png'.format(index*batch_size+i))

            plt.subplot(121)
            plt.imshow(image)
            plt.title(str(labels.numpy()[i]))
            plt.subplot(122)
            plt.imshow(images_h5[index*batch_size+i])
            plt.title(labels_h5[index*batch_size+i])
            plt.savefig(image_path)
            plt.close()
    pass

if __name__ == '__main__':
    data_root = '/root/userfolder/AneNet/Dataset/'
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    make_random_data(data_root)
    make_dataset_cla(data_root)