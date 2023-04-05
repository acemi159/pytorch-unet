from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import pickle as pkl
from os.path import join

import cv2 as cv

DATASET_ROOT_DIR = "data/ADE20K_2021_17_01/"
image_transforms = {
    'rgb' : transforms.Compose([
                                transforms.Resize((512,512)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]),
    'grayscale' : transforms.Compose([
                                transforms.Resize((512, 512)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5), std=(0.5))
    ])
}

class ADE20K(Dataset):
    def __init__(self, to_train=True):
        self.root_dir = DATASET_ROOT_DIR
        self.index_file = join(self.root_dir, 'index_ade20k.pkl')
        self.train_dir = join(self.root_dir, 'images/ADE/training/')
        self.valid_dir = join(self.root_dir, 'images/ADE/validation/')

        # Parse dataset information from index file to corresponding variables
        self.filenames = None
        self.folders = None
        self.scenes = None
        self.objectIsParts = None
        self.objectPresences = None
        self.objectCounts = None
        self.objectNames = None
        self.ExtractIndices()
        print(f"n_filenames: {len(self.filenames)}-- n_folders: {len(self.folders)}-- n_scenes: {len(self.scenes)}--- n_isParts: {len(self.objectIsParts)}\nn_objectPresences: {len(self.objectPresences)}--- n_objectCounts: {len(self.objectCounts)}--- n_objectnames: {len(self.objectNames)}")


        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def PrintDescription(self):
        with open(self.index_file, 'rb') as file:
            index_dataset = pkl.load(file)

        for attribute, description in index_dataset['description'].items():
            print(f"* {attribute} ::::::    {description}")

    def ExtractIndices(self):
        with open(self.index_file, 'rb') as file:
            data = pkl.load(file)

        self.filenames = data['filename']
        self.folders = data['folder']
        self.scenes = data['scene']
        self.objectIsParts = data['objectIsPart']
        self.objectPresences = data['objectPresence']
        self.objectCounts = data['objectcounts']
        self.objectNames = data['objectnames']

dataset = ADE20K()
print("FILENAME index 0 : ", dataset.filenames[0])
print("SCENE index 0 : ", dataset.scenes[0])
print("FOLDER index 0 : ", dataset.folders[0])
print("OBJECTISPART index 0 : ", dataset.objectIsParts[0])
print("OBJECTPRESENCE index 0 : ", dataset.objectPresences[0])
print("OBJECTCOUNT index 0 : ", dataset.objectCounts[0])
print("OBJECTNAMES index 0 : ", dataset.objectNames[0])