from PIL import Image
import numpy as np
#import torch

BUCKET_SIZE = 10

def load_tif(tif_path, gt_path):
    """
    Функция загрузки tif.
    
    Возвращает список images и masks
    """
    
    tif_path = tif_path.replace("'", "")
    gt_path = gt_path.replace("'", "")
    
    images = []
    masks = []
    try:
        image = Image.open(tif_path)
        mask = Image.open(gt_path)
    except FileNotFoundError:
        return None, None
    
    i = 0
    while True:
        try:
            mask.seek(i)
            image.seek(i)
          
            mask_pil = mask.convert('L')
            image_pil = image.convert('RGB')
            
            cur_mask = np.array(mask_pil)
            
            if len(np.unique(np.concatenate(cur_mask, axis=0))) > 1:
                masks.append(mask_pil)
                images.append(image_pil)
            i += 1
        except EOFError:
            break
    return images, masks
            
def get_tifs_and_masks(tif_pathes, gt_pathes):
    
    images = []
    masks = []
    
    cnt = 0
    sz = len(tif_pathes)
    for tif_path, gt_path in zip(tif_pathes, gt_pathes):
        print(f"Loading data: {cnt} / {sz}")
        cnt += 1
        
        image, mask = load_tif(tif_path, gt_path)   
        if image is None:
            continue
        
        while len(image) >= BUCKET_SIZE:
            images.append(image[:BUCKET_SIZE])
            masks.append(mask[:BUCKET_SIZE])
            image = image[BUCKET_SIZE:]
            mask = mask[BUCKET_SIZE:]
        
        if len(image) > 0:
            images.append(image)
            masks.append(mask)
            
    return images, masks
         