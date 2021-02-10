import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from UCF50 import UCF50
from setting import cfg_data 
import torch
import torch.nn.functional as F
import random
import numpy as np

def get_min_size(batch):

    min_ht = cfg_data.TRAIN_SIZE[0]
    min_wd = cfg_data.TRAIN_SIZE[1]

    for i_sample in batch:
        
        _,ht,wd = i_sample.shape
        if ht<min_ht:
            min_ht = ht
        if wd<min_wd:
            min_wd = wd
    return min_ht,min_wd

def image_crop(image,p_size):
    w,h = image.shape
    new_h = int(np.ceil(h/p_size)*p_size)
    new_w = int(np.ceil(w/p_size)*p_size)

    return new_x
def image_train_crop(img,den1,den2,deni,p_size):
    
    # we retreive the highest probable x and y via the density
    h,w = den1.size()
    indexes = [i for i in range(w*h)]
    # we use density k=1 to get more spread probabilities
    prob = torch.flatten(den1/torch.sum(den1))
    h,w = den1.shape
    index = np.random.choice(indexes,p=prob)
    x,y = np.unravel_index(index,[h,w])
    
    wy = min(y+p_size,w-1)
    hx = min(x+p_size,h-1)
    
    imgc  = img[:,x:hx,y:wy]
    den1c = den1[x:hx,y:wy]
    den2c = den2[x:hx,y:wy]
    denic = deni[x:hx,y:wy]
    
    pad_right = 0
    pad_bot = 0
    pad_bot = (y+p_size)-w
    pad_right = (x+p_size)-h

    if pad_bot>0 or pad_right>0:
        pad_right = max(pad_right+1,0)
        pad_bot = max(pad_bot+1,0)
        imgc = F.pad(input=imgc, pad=(0, pad_bot, 0, pad_right), mode='constant', value=0)
        den1c = F.pad(input=den1c, pad=(0, pad_bot, 0, pad_right), mode='constant', value=0)
        den2c = F.pad(input=den2c, pad=(0, pad_bot, 0, pad_right), mode='constant', value=0)
        denic = F.pad(input=denic, pad=(0, pad_bot, 0, pad_right), mode='constant', value=0)
    
    # We have re-normalize all the maps to get the right count
    denic /= denic.max()
    gt_count = denic.sum()
    den1c = (den1c/den1c.sum())*gt_count
    den2c = (den2c/den2c.sum())*gt_count
    return imgc, den1c, den2c, denic

def share_memory(batch):
    out = None
    if False:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)
    return out

class SHHA_collate_custom(object):
    def __init__(self, p_size):
        self.p_size = p_size
    def __call__(self, batch):
        return self.SHHA_collate(batch,self.p_size)
    def SHHA_collate(self,batch,p_size):
        # @GJY 
        r"""Puts each data field into a tensor with outer dimension batch size"""

        transposed = list(zip(*batch)) # imgs and dens
        imgs, den1s, den2s, denis = [transposed[0],transposed[1],transposed[2],transposed[3]]
        
        print(denis[0].sum())

        error_msg = "batch must contain tensors; found {}"
        if isinstance(imgs[0], torch.Tensor) and isinstance(den1s[0], torch.Tensor) and isinstance(den2s[0], torch.Tensor) and isinstance(denis[0], torch.Tensor):

            min_ht, min_wd = get_min_size(imgs)

            cropped_imgs = []
            cropped_den1s = []
            cropped_den2s = []
            cropped_denis = []

            for i_sample in range(len(batch)):
                _img, _den1, _den2, _deni =  image_train_crop(imgs[i_sample],den1s[i_sample],den2s[i_sample],denis[i_sample],p_size)
                cropped_imgs.append(_img)
                cropped_den1s.append(_den1)
                cropped_den2s.append(_den2)
                cropped_denis.append(_deni)
                print(_deni.sum())
            
            cropped_imgs = torch.stack(cropped_imgs, 0, out=share_memory(cropped_imgs))
            cropped_den1s = torch.stack(cropped_den1s, 0, out=share_memory(cropped_den1s))
            cropped_den2s = torch.stack(cropped_den2s, 0, out=share_memory(cropped_den2s))
            cropped_denis = torch.stack(cropped_denis, 0, out=share_memory(cropped_denis))
            
            return [cropped_imgs,cropped_den1s,cropped_den2s,cropped_denis]

        raise TypeError((error_msg.format(type(batch[0]))))

def SHHA_collate(batch):
    # @GJY 
    r"""Puts each data field into a tensor with outer dimension batch size"""

    transposed = list(zip(*batch)) # imgs and dens
    imgs, den1s, den2s, denis = [transposed[0],transposed[1],transposed[1],transposed[2]]


    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(den1s[0], torch.Tensor) and isinstance(den2s[0], torch.Tensor) and isinstance(denis[0], torch.Tensor):

        min_ht, min_wd = get_min_size(imgs)

        cropped_imgs = []
        cropped_den1s = []
        cropped_den2s = []
        cropped_denis = []

        for i_sample in range(len(batch)):
            _img, _den1, _den2, _deni =  image_train_crop(imgs[i_sample],den1s[i_sample],den2s[i_sample],denis[i_sample],224)
            cropped_imgs.append(_img)
            cropped_den1s.append(_den1)
            cropped_den2s.append(_den2)
            cropped_denis.append(_deni)

        cropped_imgs = torch.stack(cropped_imgs, 0, out=share_memory(cropped_imgs))
        cropped_den1s = torch.stack(cropped_den1s, 0, out=share_memory(cropped_den1s))
        cropped_den2s = torch.stack(cropped_den2s, 0, out=share_memory(cropped_den2s))
        cropped_denis = torch.stack(cropped_denis, 0, out=share_memory(cropped_denis))

        return [cropped_imgs,cropped_den1s,cropped_den2s,cropped_denis]

    raise TypeError((error_msg.format(type(batch[0]))))


def get_train_folder(val_folder):
    all_folder = [1,2,3,4,5]
    del all_folder[val_folder-1]

    return all_folder

def loading_data():
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    train_main_transform = None
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
    
    SHHA_collate_112 = SHHA_collate_custom(112)
    SHHA_collate_224 = SHHA_collate_custom(224)
    SHHA_collate_448 = SHHA_collate_custom(448)

    val_folder = cfg_data.VAL_INDEX
    train_folder = get_train_folder(val_folder)

    train_set = UCF50(cfg_data.DATA_PATH, [""], 'train',main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    train_loader = DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=8, collate_fn=SHHA_collate_224, shuffle=True, drop_last=True)
    
    val_set = UCF50(cfg_data.DATA_PATH, [""], 'test', main_transform=None, img_transform=img_transform, gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False)

    return train_loader, val_loader, restore_transform
