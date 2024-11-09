#%%writefile /kaggle/working/research/loader/data_loader_tiff.py
# %load /kaggle/working/research/loader/tnscui_utils
import random
from PIL import Image
import sys
import numpy as np

sys.path.append('/kaggle/working/research/loader/tnscui_utils/')

from tnscui_utils.TNSUCI_util import char_color
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F

from img_mask_aug import data_aug


class TifFolder(data.Dataset):

    def __init__(self, images, masks, image_size=512, mode='train', augmentation_prob=0.4):

        super().__init__()
        self.tiff_images = images
        self.tiff_masks = masks
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270, 45, 135, 215, 305]
        self.augmentation_prob = augmentation_prob
        self.resize_range = [520, 560]
        self.CropRange = [400, 519]

    def __getitem__(self, index):

        tiff_images = self.tiff_images[index]
        tiff_masks = self.tiff_masks[index]

        Transform = []
        Transform_GT = []

        Transform.append(
            T.Resize((self.image_size, self.image_size), interpolation=Image.BICUBIC))
        Transform_GT.append(
            T.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST))
        Transform.append(T.ToTensor())
        Transform_GT.append(T.ToTensor())

        Transform = T.Compose(Transform)
        Transform_GT = T.Compose(Transform_GT)

        tiff_images_prcd = []
        tiff_masks_prcd = []

        for image, mask in zip(tiff_images, tiff_masks):
            tiff_images_prcd.append(Transform(image))
            tiff_masks_prcd.append((Transform_GT(mask) > 0).to(torch.int64))

        images_concated = None
        masks_concated = None

        for image, mask in zip(tiff_images_prcd, tiff_masks_prcd):
            image = image.unsqueeze(1)
            mask = mask.unsqueeze(1)
            if images_concated is None:
                images_concated = image
                masks_concated = mask
            else:
                images_concated = torch.cat((images_concated, image), dim=1)
                masks_concated = torch.cat((masks_concated, mask), dim=1)

        return images_concated, masks_concated

    def __len__(self):

        return len(self.tiff_images)


class TifFolder_difficult(data.Dataset):

    def __init__(self, images, masks, image_size=512, mode='train', augmentation_prob=0.4):

        super().__init__()
        self.tiff_images = images
        self.tiff_masks = masks
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270, 45, 135, 215, 305]
        self.augmentation_prob = augmentation_prob
        self.resize_range = [520, 560]
        self.CropRange = [400, 519]

    def __getitem__(self, index):

        tiff_images = self.tiff_images[index]
        tiff_masks = self.tiff_masks[index]

        Transform = []
        Transform_GT = []

        np_seed = random.randint(1, 100)
        imgaug_seed = random.randint(1, 100)

        p_transform = random.random()

        tiff_images_prcd_1 = []
        tiff_masks_prcd_1 = []
        
        #print(f"TYPE TIFF IMAGES: {type(tiff_images[0])}")

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            """
            for image, mask in zip(tiff_images, tiff_masks):
                image, mask = data_aug(image, mask, np_seed, imgaug_seed)
                image = Image.fromarray(image)
                GT = Image.fromarray(GT)
                tiff_images_prcd_1.append(image)
                tiff_masks_prcd_1.append(mask)
            """
            images_aug, masks_aug = data_aug(tiff_images, tiff_masks)
            for img, mask in zip(images_aug, masks_aug):
                tiff_images_prcd_1.append(Image.fromarray(img))
                tiff_masks_prcd_1.append(Image.fromarray(mask))
            
            #print(f"PROCESSED TYPE: {type(tiff_images_prcd_1[0])}")
            
        if len(tiff_images_prcd_1) == 0:
            tiff_images_prcd_1 = tiff_images
            tiff_masks_prcd_1 = tiff_masks

        final_size = self.image_size
        Transform.append(T.Resize((final_size, final_size),
                                  interpolation=Image.BICUBIC))
        Transform_GT.append(
            T.Resize((final_size, final_size), interpolation=Image.NEAREST))

        Transform.append(T.ToTensor())
        Transform_GT.append(T.ToTensor())

        Transform = T.Compose(Transform)
        Transform_GT = T.Compose(Transform_GT)

        tiff_images_prcd = []
        tiff_masks_prcd = []

        for image, mask in zip(tiff_images_prcd_1, tiff_masks_prcd_1):
            tiff_images_prcd.append(Transform(image).unsqueeze(1))
            tiff_masks_prcd.append((Transform_GT(mask) > 0).to(torch.int32).unsqueeze(1))
            
        images_concated = torch.cat(tiff_images_prcd, dim=1)
        masks_concated = torch.cat(tiff_masks_prcd, dim=1)
        """
        for image, mask in zip(tiff_images_prcd, tiff_masks_prcd):
            image = image.unsqueeze(1)
            mask = mask.unsqueeze(1)
            if images_concated is None:
                images_concated = image
                masks_concated = mask
            else:
                images_concated = torch.cat((images_concated, image), dim=1)
                masks_concated = torch.cat((masks_concated, mask), dim=1)
        """

        return images_concated, masks_concated
        

    def __len__(self):

        return len(self.tiff_images)


def get_loader(tiffs_images, tiffs_masks,
               image_size=512, batch_size=4, num_workers=2, mode='train', augmentation_prob=0.4):

    dataset = TifFolder(tiffs_images, tiffs_masks,
                        image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True
                                  )
    return data_loader


def get_loader_difficult(tiffs_images, tiffs_masks,
                         image_size=512,  batch_size=1, num_workers=2, mode='train', augmentation_prob=0.4):

    dataset = TifFolder_difficult(tiffs_images, tiffs_masks,
                                  image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True
                                  )
    return data_loader, dataset
