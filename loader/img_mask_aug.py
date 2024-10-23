import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import imgaug.random as iarandom

"""
Данный модуль используется для аугментации в классе Tif_Folder_difficult
"""

def data_aug(imgs, masks, np_seed=42, imgaug_seed = 42): 
    
    """
    Делает аугментацию изображений и масок.
    
    Возвращает аугментированные результаты.
    Аугментация проводится без изменения размера входного изображения
    """
        
    iarandom.seed(imgaug_seed)
    np.random.seed(np_seed)
    
    imgs = np.array(imgs)
    masks =  np.array(masks).astype(np.uint8)
    
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    
    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5), # Отрафжение слева направо
            iaa.Flipud(0.5), # Отражение сверху вниз
            sometimes(iaa.Crop(percent=(0, 0.1))), # Обрезаем
            sometimes(iaa.Affine(                    # Применяем афинное преобразование
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},   
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},    
                rotate=(-45, 45),  
                shear=(-16, 16), 
                order=[0, 1], 
                cval=(0, 255),
                mode=ia.ALL,   
            )),
            iaa.SomeOf((0, 5), # случайно выбирает от 0 до 5 аугментаций из данного списка
                [
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.25)), # Контрастность
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                    iaa.contrast.LinearContrast((0.75, 1.25), per_channel=0.5),
                    iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                               iaa.MedianBlur(k=(3, 11)),
                           ]),
                ]         
            ),
            iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                           ),
            iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout(
                                   (0.03, 0.15), size_percent=(0.02, 0.05),
                                   per_channel=0.2
                               ),
                           ]),
            sometimes(
                               iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                           ),
        ],
        random_order=True
    )
    
    images_aug = []
    masks_aug = []
    seq_det = seq.to_deterministic()
    for image, mask in zip(imgs, masks):
        images_aug.append(
            seq_det.augment_image(image)
        )
        segmap = ia.SegmentationMapsOnImage(mask, shape=mask.shape)
        masks_aug.append(
            seq_det.augment_segmentation_maps(segmap).get_arr().astype(np.uint8)
        )
        
    return  images_aug, masks_aug
