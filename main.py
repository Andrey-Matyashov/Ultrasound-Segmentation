import numpy as np
from sklearn.model_selection import train_test_split
from loader.tif_loader import get_tifs_and_masks
from pathes_parser.pathes_parser import get_pathes
from loader.data_loader_tiff import get_loader
from Trainer.trainer import get_trained_model

PATHES = '/home/andmats/mephi/research/pathes.txt'

if __name__ == '__main__':  
    
    np.random.seed(42)
    
    tif_pathes, gt_pathes = get_pathes(PATHES)
    
    tif_pathes_train, tif_pathes_test, gt_pathes_train, gt_pathes_test = train_test_split(
        tif_pathes, gt_pathes, test_size=0.3
    )
    
    tifs_train, gts_train = get_tifs_and_masks(tif_pathes_train, gt_pathes_train)
    tifs_test, gts_test = get_tifs_and_masks(tif_pathes_test, gt_pathes_test)
    
    train_loader = get_loader(tifs_train, gts_train)
    test_loader = get_loader(tifs_test, gts_test)
    
    model = get_trained_model(train_loader, test_loader)