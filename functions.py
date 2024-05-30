## Modules
import iio
import numpy as np
from skimage.metrics import structural_similarity
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

##Functions 

def psnr(img1, img2, peak=1):
    '''
    Computes the PSNR 'metric' between two images assumed to be in the range [0,1]
    '''
    x = (img1-img2).detach().cpu().numpy().squeeze().flatten()
    return (10*np.log10(peak**2 / np.mean(x**2)))



def reads_image(path, im_range, device):
    image = iio.read(path)
    image = image / im_range
    image = image.transpose(2,0,1)
    image = torch.Tensor(image).to(device)
    return image

ssim = lambda x, y : structural_similarity(x.detach().cpu().numpy().squeeze().transpose(1,2,0),y.detach().cpu().numpy().squeeze().transpose(1,2,0), data_range=1, channel_axis=2)
