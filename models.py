import torch
import torch.nn as nn
import numpy as np


class PixelUnshuffle3d(nn.Module):

    def __init__(self, upscale_factor):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(upscale_factor)
        self.upscale_factor = 2

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # B, C, D, H, W -> B, D, C, H, W
        return self.pixel_unshuffle(x).permute(0, 2, 1, 3, 4)


class PixelShuffle3d(nn.Module):

    def __init__(self, downscale_factor):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(downscale_factor)
        self.downscale_factor = 2

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # B, C, D, H, W -> B, D, C, H, W
        return self.pixel_shuffle(x).permute(0, 2, 1, 3, 4)


class M2MResblock(nn.Module):

    def __init__(self, num_features_in, num_features_out, normalization='none',
                 intraframe=True):
        super(M2MResblock, self).__init__()
        self.num_features_in  = num_features_in
        self.num_features_out = num_features_out
        self.normalization = normalization

        nfi = num_features_in
        nf  = num_features_out
        self.input_conv = nn.Conv3d(nfi, nf, kernel_size=(1, 1, 1))

        if normalization == 'batch':
            Normalization = nn.BatchNorm3d
        else:
            Normalization = nn.Identity

        if intraframe:
            self.conv_block = nn.Sequential(
                nn.Conv3d(nf, nf, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                Normalization(nf),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(nf, nf, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                Normalization(nf),
                nn.LeakyReLU(inplace=True))
        else:
            self.conv_block = nn.Sequential(
                nn.Conv3d(nf, nf, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(nf, nf, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                Normalization(nf),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(nf, nf, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(nf, nf, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
                Normalization(nf))

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.01,
                                    nonlinearity='leaky_relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = self.input_conv(x)
        return x + self.conv_block(x)


class M2MSubnet(nn.Module):

    def __init__(self, num_input_frames, io_channels, num_features,
                 normalization='none', intraframe=True):
        super(M2MSubnet, self).__init__()
        self.num_input_frames = num_input_frames
        self.io_channels = io_channels
        self.num_features = num_features
        self.intraframe = intraframe
        self.normalization = normalization
       
        nf = self.num_features
        ch = self.io_channels

        if normalization == 'batch':
            Normalization = nn.BatchNorm3d
        else:
            Normalization = nn.Identity

        self.input_block = nn.Sequential(
            nn.Conv3d(ch, nf, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
            M2MResblock(nf, nf, normalization, intraframe))

        self.down_block_2 = nn.Sequential(
            PixelUnshuffle3d(2),
            nn.Conv3d(4*nf, nf, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True),
            M2MResblock(nf, nf, normalization, intraframe))

        self.down_block_4 = nn.Sequential(
            PixelUnshuffle3d(2),
            nn.Conv3d(4*nf, nf, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True),
            M2MResblock(nf, nf, normalization, intraframe))

        self.bottleneck = nn.Sequential(
            PixelUnshuffle3d(2),
            nn.Conv3d(4*nf, nf, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True),
            M2MResblock(nf, nf, normalization, intraframe),
            nn.Conv3d(nf, 4*nf, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True),
            PixelShuffle3d(2))

        self.up_block_4 = nn.Sequential(
            M2MResblock(2*nf, nf, normalization, intraframe),
            nn.Conv3d(nf, 4*nf, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True),
            PixelShuffle3d(2))

        self.up_block_2 = nn.Sequential(
            M2MResblock(2*nf, nf, normalization, intraframe),
            nn.Conv3d(nf, 4*nf, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True),
            PixelShuffle3d(2))

        if intraframe:
            self.output_block = nn.Sequential(
                M2MResblock(2*nf, nf, normalization, intraframe),
                nn.Conv3d(nf, nf, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                Normalization(nf),
                nn.LeakyReLU(inplace=True))
        else:
            self.output_block = nn.Sequential(
                M2MResblock(2*nf, nf, normalization, intraframe),
                nn.Conv3d(nf, nf, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(nf, nf, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                Normalization(nf),
                nn.LeakyReLU(inplace=True))

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.01,
                                    nonlinearity='leaky_relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x0):
        x0 = self.input_block(x0)
        x2 = self.down_block_2(x0)
        x4 = self.down_block_4(x2)
        x = self.bottleneck(x4)
        x = self.up_block_4(torch.cat((x, x4), dim=1))
        x = self.up_block_2(torch.cat((x, x2), dim=1))
        return self.output_block(torch.cat((x, x0), dim=1))


class M2Mnet_rec(nn.Module):
    """ Definition of the M2Mnet model.
    Inputs of forward():
    xn: input frames of dim [N, C=5x3, H, W],
    """

    def __init__(self, num_input_frames=5, num_features=64, normalization='none'):
        super(M2Mnet_rec, self).__init__()
        self.num_input_frames = num_input_frames
        self.num_features = num_features
        self.io_channels = 3
        self.normalization = normalization

        nf = self.num_features
        ch = 2*self.io_channels

        self.intraframe_net = M2MSubnet(num_input_frames, ch, nf,
                                        normalization, intraframe=True)
        self.interframe_net = M2MSubnet(num_input_frames, ch, nf,
                                        normalization, intraframe=False)

        if normalization == 'batch':
            Normalization = nn.BatchNorm3d
        else:
            Normalization = nn.Identity

        self.merge = nn.Sequential(
            nn.Conv3d(2*nf, nf, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(nf, nf, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            Normalization(nf),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(nf, nf, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(nf, ch//2, kernel_size=(1, 3, 3), padding=(0, 1, 1)))

        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.01,
                                    nonlinearity='leaky_relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, previous_denoised_frame, x):
        '''Args:
        previous_denoised_frame: Tensor, [N, 1*C, H, W] in the [0., 1.] range
        x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
        '''

        s = x.shape

        v = 8 * np.ceil(s[2]/8).astype('int') - s[2]
        h = 8 * np.ceil(s[3]/8).astype('int') - s[3]
        p = [h//2, h - h//2, v//2, v - v//2]  # horiz and vertical padding

        y = torch.nn.functional.pad(previous_denoised_frame, p, mode='reflect')\
            .reshape(s[0], 1, self.io_channels, s[2] + v, s[3] + h)\
            .permute(0, 2, 1, 3, 4)  # [batch, channel, frame, height, width]


        x = torch.nn.functional.pad(x, p, mode='reflect')\
            .reshape(s[0], self.num_input_frames, self.io_channels, s[2] + v, s[3] + h)\
            .permute(0, 2, 1, 3, 4)  # [batch, channel, frame, height, width]


        X = torch.zeros((s[0], 2*self.io_channels, self.num_input_frames, s[2]+v, s[3]+h)).to(x.device)
        for k in range(self.num_input_frames):
            X[:,:,k,:,:] = torch.cat((x[:,:,k,:,:], y[:,:,0,:,:,]), dim=1)

        
        sp = x.shape

        return (x + self.merge(torch.cat((self.intraframe_net(X),
                                          self.interframe_net(X)), dim=1)
                               )).permute(0, 2, 1, 3, 4)\
                               [:, :, :, p[2]:sp[3]-p[3], p[0]:sp[4]-p[1]].reshape(s)


class M2Mnet_non_rec(nn.Module):
    """ Definition of the M2Mnet model.
    Inputs of forward():
    xn: input frames of dim [N, C=5x3, H, W],
    """

    def __init__(self, num_input_frames=5, num_features=64, normalization='none'):
        super(M2Mnet_non_rec, self).__init__()
        self.num_input_frames = num_input_frames
        self.num_features = num_features
        self.io_channels = 3
        self.normalization = normalization

        nf = self.num_features
        ch = self.io_channels

        self.intraframe_net = M2MSubnet(num_input_frames, ch, nf,
                                        normalization, intraframe=True)
        self.interframe_net = M2MSubnet(num_input_frames, ch, nf,
                                        normalization, intraframe=False)

        if normalization == 'batch':
            Normalization = nn.BatchNorm3d
        else:
            Normalization = nn.Identity

        self.merge = nn.Sequential(
            nn.Conv3d(2*nf, nf, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(nf, nf, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            Normalization(nf),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(nf, nf, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(nf, ch, kernel_size=(1, 3, 3), padding=(0, 1, 1)))

        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.01,
                                    nonlinearity='leaky_relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        '''Args:
        x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
        '''
        s = x.shape

        v = 8 * np.ceil(s[2]/8).astype('int') - s[2]
        h = 8 * np.ceil(s[3]/8).astype('int') - s[3]
        p = [h//2, h - h//2, v//2, v - v//2]  # horiz and vertical padding

        x = torch.nn.functional.pad(x, p, mode='reflect')\
            .reshape(s[0], self.num_input_frames, self.io_channels, s[2] + v, s[3] + h)\
            .permute(0, 2, 1, 3, 4)  # [batch, channel, frame, height, width]

        sp = x.shape
        return (x + self.merge(torch.cat((self.intraframe_net(x),
                                          self.interframe_net(x)), dim=1)
                               )).permute(0, 2, 1, 3, 4)\
                               [:, :, :, p[2]:sp[3]-p[3], p[0]:sp[4]-p[1]].reshape(s)
