from ade_dataloader import ADEDataset, TRAIN_IMAGE_PATHS, VAL_IMAGE_PATHS, DATASET_ROOT_PATH

from torch.utils.data import DataLoader
from tqdm import tqdm


trainset = ADEDataset(root_dir=VAL_IMAGE_PATHS)
trainloader = DataLoader(dataset=trainset, batch_size=8, shuffle=True, num_workers=6)

print(len(trainloader))
sth = 5

"""for i, (img, seg) in enumerate(trainloader):
    print(f"img shape : {img.shape} ----- seg shape : {seg.shape}")
"""