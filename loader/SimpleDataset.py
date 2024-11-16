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

class TifDataset(data.Dataset):
    
    def __init__(self, images, masks, image_size=512, mode='train', augmentation_prob=1):

        super().__init__()
        self.images = []
        self.masks = []
        
        for image, mask in zip(images, masks):
            image, mask = self.process_image_and_mask(image, mask, mode, augmentation_prob, image_size)
            self.images.append(image)
            self.masks.append(mask)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.masks[index]
    
    @staticmethod
    def process_image_and_mask(image, mask, mode, augmentation_prob, image_size=512):
        Transform = []
        Transform_GT = []
        
        p_transform = random.random()
        
        image_processed = None
        mask_processed = None
        
        if mode == 'train' and p_transform <= augmentation_prob:
            image, mask = data_aug([image], [mask])
            image_processed = Image.fromarray(image[0])
            mask_processed = Image.fromarray(mask[0])
        else:
            image_processed = image
            mask_processed = mask
            
        Transform.append(
            T.Resize((image_size, image_size), interpolation=Image.BICUBIC))
        Transform_GT.append(
            T.Resize((image_size, image_size), interpolation=Image.NEAREST))
        Transform.append(T.ToTensor())
        Transform_GT.append(T.ToTensor())
        
        Transform = T.Compose(Transform)
        Transform_GT = T.Compose(Transform_GT)
        
        image_processed = Transform(image_processed)
        mask_processed = (Transform_GT(mask_processed) > 0).to(torch.int64)
        
        return image_processed, mask_processed
    
def get_simple_loader(images, masks, image_size=512, mode='train', augmentation_prob=1, batch_size=128):
    all_images = []
    all_masks = []
    for image, mask in zip(images, masks):
        all_images.extend(image)
        all_masks.extend(mask)
    dataset = TifDataset(all_images, all_masks, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=128,
                                  shuffle=True,
                                  num_workers=0)
    return data_loader