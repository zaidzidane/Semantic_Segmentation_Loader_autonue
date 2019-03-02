import os
import torch
import numpy as np
import scipy.misc as m
import cv2

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *
import matplotlib.pyplot as plt


class cityscapesLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """


    colors = [ #[  0,   0,   0],
        [ 70, 130, 180],
        [220,  20,  60],
          # [128,   0, 128],
          # [255, 0,   0],
          # [ 0,   0,  60],
          # [0,  60, 100],
          # [ 0,   0, 142],
          # [119,  11,  32],
          # [244,  35, 232],
          # [  0,   0, 160],
          # [153, 153, 153],
          # [220, 220,   0],
          # [250, 170,  30],
          # [102, 102, 156],
          # [128,   0,   0],
          # [128,  64, 128],
          # [238, 232, 170],
          # [190, 153, 153],
          # [  0,   0, 230],
          # [128, 128,   0],
          # [128,  78, 160],
          # [150, 100, 100],
          # [255, 165,   0],
          # [180, 165, 180],
          # [107, 142,  35],
          # [201, 255, 229],
          # [0,   191, 255],
          # [ 51, 255,  51],
          # [250, 128, 114],
          # [127, 255,   0],
          # [255, 128,   0],
          # [  0, 255, 255],
          # [178, 132, 190],
          # [128, 128,  64],
          # [102,   0, 204],
          # #[  0, 153, 153],
          #[255, 255, 255],
        ]


    




    label_colours = dict(zip(range(35), colors))
    print(label_colours)

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        augmentations=None,
        img_norm=True,
        version="cityscapes",
        test_mode=False,

    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations 
        """
        self.root = root
        #print(root)
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
        self.mean = np.array(self.mean_rgb[version])
        print(self.mean)
        self.files = {}
        #self.q="/home/zaid/Documents/mask_rcnn_pytorvh/pytorch-semseg/gtFine/train_aug/"
        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".jpg")

        self.void_classes = [35,36,37,38,39]
        # self.valid_classes = [
        #     1,
        #     2,
        #     3,
        #     4,
        #     5,
        #     6,
        #     7,
        #     8,
        #     9,
        #     10,
        #     11,
        #     12,
        #     13,
        #     14,
        #     15,
        #     16,
        #     17,
        #     18,
        #     19,
        #     20,
        #     21,
        #     22,
        #     23,
        #     24,
        #     25,
        #     26,
        #     27,
        #     28,
        #     29,
        #     30,
        #     31,
        #     32,
        #     33,
        #     34,
        #     35,

        # ]
        self.valid_classes = [
            #16 ,   
            #90 ,
            0, 
            1 , 
            2 , 
            3 , 
            4 , 
            5 , 
            6 , 
            7 , 
            8 , 
            9 , 
            10 , 
            11 , 
            12 , 
            13 , 
            14 , 
            15 , 
            16 , 
            17 , 
            18, 
            19 , 
            20 , 
            21 , 
            22 , 
            23 , 
            24 , 
            25 , 
            26 , 
            27 , 
            28 , 
            29 , 
            30 , 
            31 , 
            32 , 
            33 ,
            34,
            35,
            36,
            37,
            38,
            39,  
        ]
        self.class_names = [ 
        's_w_d', 
        's_y_d', 
     #  'ds_w_dn', 
     #  'ds_y_dn', 
     #  'sb_w_do', 
     #  'sb_y_do', 
     #    'b_w_g', 
     #    'b_y_g', 
     #   'db_w_g', 
     #   'db_y_g', 
     #   'db_w_s', 
     #    's_w_s', 
     #   'ds_w_s',
     #    's_w_c', 
     #    's_y_c', 
     #    's_w_p', 
     #    's_n_p',
     #   'c_wy_z',
     #    'a_w_u',
     #    'a_w_t', 
     #   'a_w_tl', 
     #   'a_w_tr',
     #  'a_w_tlr', 
     #    'a_w_l', 
     #    'a_w_r', 
     #   'a_w_lr', 
     #   'a_n_lu', 
     #   'a_w_tu', 
     #    'a_w_m', 
     #    'a_y_t', 
     #   'b_n_sr', 
     #  'd_wy_za', 
     #  'r_wy_np',
     # 'vom_wy_n', 
     #   'om_n_n', 
        ]



        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(35)))

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        


        img_path = self.files[self.split][index].rstrip()
        #print(img_path)
        lbl_path = os.path.join(
            self.annotations_base,
            os.path.basename(img_path)[:-4] + "_bin.png",
             )
        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        print(img.shape)
        plt.imshow(img)
        plt.show() 
        lbl = m.imread(lbl_path,flatten=1)
        #lbl=lbl[:,:,1]
        plt.imshow(img,cmap='gray')
        plt.show() 
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1])
        )  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
         
        classes = np.unique(lbl)
        print(classes)
        lbl = lbl.astype(float)
        #print(self.img_size[0],self.img_size[1])
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        plt.imshow(lbl)
        plt.show()
        

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")
        
        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
           
            mask[mask == _voidc] = self.ignore_index

        for _validc in self.valid_classes:
           
            mask[mask == _validc] = self.class_map[_validc]
        return mask


if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = "/datasets01/cityscapes/112817/"
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        import pdb;pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = raw_input()
        if a == "ex":
            break
        else:
            plt.close()
