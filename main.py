import os
from os.path import dirname, join, realpath
import argparse
import numpy as np

import torch
import torch.nn as nn

import iio

from functions import *

ROOT = dirname(realpath(__file__))

def main():
    # Options
    parser = argparse.ArgumentParser()
    parser.add_argument("--input"      , type=str, default=""            , help='path to input frames (C type).'                      )
    parser.add_argument("--output"     , type=str, default="./%03d.png"  , help='path to output image (C type).'                      )
    parser.add_argument("--first"      , type=int, default=1             , help='index first frame.'                                  )
    parser.add_argument("--last"       , type=int, default=14            , help='index last frame.'                                   )
    parser.add_argument("--im_range"   , type=int, default=255           , help='range of the data.'                                  )
    parser.add_argument("--nb_frames"  , type=int, default=7             , help='number of frames taken as input by the network.'     )
    parser.add_argument("--nb_features", type=int, default=48            , help='number of features of the first conv in the network.')
    parser.add_argument("--recurrence" , type=str, default='rec'         , help='choose between recurrent or non recurrent network.'  )
    parser.add_argument('--add_noise'  , type=str, default="true"                                                                     )
    parser.add_argument("--sigma"      , type=float, default=25          , help='noise level (in range [0,255]).'                     )
    args = parser.parse_args()
    args.add_noise = args.add_noise.lower() == "true"

    AlgoInfoFile = open("algo_info.txt", "w")



   # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_list=[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        device = torch.device('cpu')
   

    # Initialisation
    u1 = iio.read("input_0.png")
    H, W, C = u1.shape

    # Load the model
    if args.recurrence=='recurrent':
        from models import M2Mnet_rec
        model = M2Mnet_rec(num_input_frames=args.nb_frames,num_features=args.nb_features)
        init_rec = torch.zeros((1, 3, H, W)).to(device)
        network_path = join(ROOT, 'model_rec.pth')
    else:
        from models import M2Mnet_non_rec
        model = M2Mnet_non_rec(num_input_frames=args.nb_frames,num_features=args.nb_features)
        network_path = join(ROOT, 'model_non_rec.pth')
   
    # Load saved weights
    state_dict = torch.load(network_path)
    model.load_state_dict(state_dict)
    model.to(device)
       
    model.eval()
    
    # Read frames (either noisy or clean depending on the user will)
    u1  = reads_image('input_0.png',  args.im_range, device)
    u2  = reads_image('input_1.png',  args.im_range, device)
    u3  = reads_image('input_2.png',  args.im_range, device)
    u4  = reads_image('input_3.png',  args.im_range, device)
    u5  = reads_image('input_4.png',  args.im_range, device)
    u6  = reads_image('input_5.png',  args.im_range, device)
    u7  = reads_image('input_6.png',  args.im_range, device)
    u8  = reads_image('input_7.png',  args.im_range, device)
    u9  = reads_image('input_8.png',  args.im_range, device)
    u10 = reads_image('input_9.png',  args.im_range, device)
    u11 = reads_image('input_10.png', args.im_range, device)
    u12 = reads_image('input_11.png', args.im_range, device)
    u13 = reads_image('input_12.png', args.im_range, device)
    u14 = reads_image('input_13.png', args.im_range, device)

    # Gathered the frames of the first stack in a single tensor  
    inframes = [u1, u2, u3, u4, u5, u6, u7]
    stack = torch.stack(inframes, dim=0).contiguous().view((1, args.nb_frames*C, H, W)).to(device)

    #Add noise if needed
    if args.add_noise:
        AlgoInfoFile.write("add_noise=1" + '\n') #for the choice of the IPOL gallery
        stack1_gt = stack.clone()
        stack = stack + torch.normal(0, args.sigma / args.im_range, stack.shape).to(device)

        
    # Compute result
    with torch.no_grad():
        if args.recurrence=='recurrent':
            out1 = model(init_rec, stack)
            init_rec = out1[:,-3:].clone()
        else:
                out1 = model(stack)


    # Gathered the frames of the second stack in a single tensor  
    inframes = [u8, u9, u10, u11, u12, u13, u14]
    stack = torch.stack(inframes, dim=0).contiguous().view((1, args.nb_frames*C, H, W)).to(device)

    #Add noise if needed and save the noisy middle frame
    if args.add_noise:
        stack2_gt = stack.clone()
        stack = stack + torch.normal(0, args.sigma / args.im_range, stack.shape).to(device)
        noisy = stack[0,C*args.nb_frames//2:C*(args.nb_frames//2+1)].detach().cpu().numpy().squeeze().transpose(1,2,0)
        iio.write("noisy.tiff", noisy)
        iio.write("noisy.png", noisy.round().clip(0,255).astype(np.uint8))
        
    # Compute result
    with torch.no_grad():
        if args.recurrence=='recurrent':
            out2 = model(init_rec, stack)
        else:
                out2 = model(stack)

    # If available (only if we provide clean frames and add noise), we compute the PSNR and SSIM (average on the two stacks)
    if args.add_noise:
        AlgoInfoFile.write("is_gt=1") #for the choice of the IPOL gallery
        iio.write("gt.png", u11) 
        PSNR = (psnr(stack1_gt, out1) + psnr(stack2_gt, out2)) / 2
        SSIM = (ssim(255*stack1_gt, 255*out1) + ssim(255*stack2_gt, 255*out2)) / 2
        print("Evalutation:")
        print("PSNR = {:4.2f}dB, SSIM = {:4.3f}".format(PSNR, SSIM))
    
    # Store result
    output = 255*out2[0,C*args.nb_frames//2:C*(args.nb_frames//2+1)].detach().cpu().numpy().squeeze().transpose(1,2,0)
    iio.write("output.tiff", output)
    iio.write("output.png", output.round().clip(0,255).astype(np.uint8))


    
if __name__ == '__main__':
    main()
