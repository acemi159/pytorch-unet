from ade_utils import utils_ade20k

from PIL import Image
from glob import glob
from torch.utils.data import Dataset

import pickle as pkl
import numpy as np

root = "data/"

TRAIN_IMAGE_PATHS = "data/ADE20K_2021_17_01/images/ADE/training/"
VAL_IMAGE_PATHS = "data/ADE20K_2021_17_01/images/ADE/validation/"
DATASET_ROOT_PATH = "data/ADE20K_2021_17_01/"


class ADEDataset(Dataset):
    def __init__(self, root_dir : str, transform= None):
        self.root_dir = root_dir

        self.transform = transform
        self.image_paths = glob(self.root_dir + "*/*/*.jpg")

        # Get the indices of objects
        with open(DATASET_ROOT_PATH+"index_ade20k.pkl", 'rb') as f:
            index_ade20k = pkl.load(f)
        self.obj_id = np.where(index_ade20k['objectPresence'][:])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        sample_info = utils_ade20k.loadAde20K(self.image_paths[index])

        image = Image.open(sample_info['img_name'])
        segmentation = Image.open(sample_info['segm_name'])
        
        return (image, segmentation)
