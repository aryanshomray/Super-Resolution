

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import joblib
import cv2
import os
import argparse
import math
from torch.utils.tensorboard import SummaryWriter
from metric import ssim as SSIM, PSNR



def upsample_filt(size):

    factor = (size + 1) // 2
    if (size & 1 == 1):
        center = factor - 1
    else:
        center = factor - 1/2
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(filter_size, weights):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    f_out = weights.size(0)
    f_in = weights.size(1)
    weights = np.zeros((f_out,
                        f_in,
                        4,
                        4), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(f_out):
        for j in range(f_in):
            weights[i, j, :, :] = upsample_kernel
    return torch.Tensor(weights)


CNUM = 64  # 64


class FeatureExtraction(nn.Module):
    def __init__(self, level):
        super(FeatureExtraction, self).__init__()
        if level == 1:
            self.conv0 = nn.Conv2d(1, CNUM / 4, (3, 3), (1, 1), (1, 1))  # RGB
        else:
            self.conv0 = nn.Conv2d(CNUM, CNUM / 4, (3, 3), (1, 1), (1, 1))
        self.conv1 = nn.Conv2d(CNUM / 4, CNUM / 4, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(CNUM / 4, CNUM / 2, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(CNUM / 2, CNUM / 2, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(CNUM / 2, CNUM / 2, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(CNUM / 2, CNUM / 2, (3, 3), (1, 1), (1, 1))
        self.conv6 = nn.Conv2d(CNUM / 2, CNUM, (3, 3), (1, 1), (1, 1))
        self.conv7 = nn.Conv2d(CNUM, CNUM, (3, 3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(CNUM, CNUM, (3, 3), (1, 1), (1, 1))
        self.conv9 = nn.Conv2d(CNUM, CNUM, (3, 3), (1, 1), (1, 1))
        self.conv10 = nn.Conv2d(CNUM, CNUM, (3, 3), (1, 1), (1, 1))
        self.convt_F = nn.ConvTranspose2d(CNUM, CNUM, (4, 4), (2, 2), (1, 1))
        self.LReLus = nn.LeakyReLU(negative_slope=0.1)
        self.convt_F.weight.data.copy_(
            bilinear_upsample_weights(4, self.convt_F.weight))

    def forward(self, x):
        out = self.LReLus(self.conv0(x))
        out = self.LReLus(self.conv1(out))
        out = self.LReLus(self.conv2(out))
        out = self.LReLus(self.conv3(out))
        out = self.LReLus(self.conv4(out))
        out = self.LReLus(self.conv5(out))
        out = self.LReLus(self.conv6(out))
        out = self.LReLus(self.conv7(out))
        out = self.LReLus(self.conv8(out))
        out = self.LReLus(self.conv9(out))
        out = self.LReLus(self.conv10(out))
        out = self.LReLus(self.convt_F(out))
        return out


class ImageReconstruction(nn.Module):
    def __init__(self):
        super(ImageReconstruction, self).__init__()
        self.conv_R = nn.Conv2d(CNUM, 1, (3, 3), (1, 1), (1, 1))  # RGB
        self.convt_I = nn.ConvTranspose2d(1, 1, (4, 4), (2, 2), (1, 1))  # RGB
        self.convt_I.weight.data.copy_(
            bilinear_upsample_weights(4, self.convt_I.weight))

    def forward(self, LR, convt_F):
        convt_I = self.convt_I(LR)
        conv_R = self.conv_R(convt_F)

        HR = convt_I + conv_R
        return HR


class LapSRN(nn.Module):
    def __init__(self):
        super(LapSRN, self).__init__()
        self.FeatureExtraction1 = FeatureExtraction(level=1)
        self.FeatureExtraction2 = FeatureExtraction(level=2)
        self.FeatureExtraction3 = FeatureExtraction(level=3)
        self.ImageReconstruction1 = ImageReconstruction()
        self.ImageReconstruction2 = ImageReconstruction()
        self.ImageReconstruction3 = ImageReconstruction()

    def forward(self, LR):
        convt_F1 = self.FeatureExtraction1(LR)
        HR_2 = self.ImageReconstruction1(LR, convt_F1)

        convt_F2 = self.FeatureExtraction2(convt_F1)
        HR_4 = self.ImageReconstruction2(HR_2, convt_F2)

        convt_F3 = self.FeatureExtraction3(convt_F2)
        HR_8 = self.ImageReconstruction3(HR_4, convt_F3)

        return HR_2, HR_4, HR_8

class Image_Dataset(torch.utils.data.Dataset):

    def __init__(self, train = True, filename = None):

        self.train = train

        if self.train:
            self.filenames = [os.path.join('Data', img_name) for img_name in os.listdir('Data')]
        else:
            self.filenames = [os.path.join('testdata',filename, img_name) for img_name in os.listdir(os.path.join('testdata', filename))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        if self.train:
            HR = joblib.load(self.filenames[idx])
        else:
            HR = cv2.imread(self.filenames[idx])

        width, height, _ = HR.shape
        LR = cv2.resize(HR, (width//8, height//8))
        HR_2 = cv2.resize(HR, (width//4, height//4))
        HR_4 = cv2.resize(HR, (width//2, height//2))
        
        HR = torch.tensor((HR)/255).permute([2,0,1]).float()
        LR = torch.tensor((LR)/255).permute([2,0,1]).float()

        return LR, HR_2, HR_4, HR



def train(batch_size, num_workers, epochs, lr = 1e-3, device = 'cpu', save_every = 10):
    #Tensorboard Summary Writer Initialization
    writer = SummaryWriter(log_dir=os.path.join('Logs',FILENAME))

    Dataset = Image_Dataset()
    Dataloader = torch.utils.data.DataLoader(Dataset, batch_size, False, num_workers=num_workers)

    model = LapSRN().to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr)

    writer.add_scalar('Epochs', epochs)
    writer.add_scalar('Batch_Size', batch_size)
    writer.add_scalar('LR', lr)

    for epoch in range(epochs):
        for idx, (input, target1, target2, target3) in enumerate(Dataloader):

            input = input.to(device)
            target1 = target1.to(device)
            target2 = target2.to(device)
            target3 = target3.to(device)

            if not epoch and not idx:
                writer.add_graph(model, input)

            print(idx)

            optimizer.zero_grad()
            HR1,HR2,HR3 = model(input)
            loss = F.mse_loss(HR1, target1) + F.mse_loss(HR2, target2) + F.mse_loss(HR3, target3)
            loss.backward()
            optimizer.step()


            psnr1 = PSNR(target1.detach(), HR1.detach())
            psnr2 = PSNR(target2.detach(), HR2.detach())
            psnr3 = PSNR(target3.detach(), HR3.detach())

            ssim1 = SSIM(target1.detach(), HR1.detach())
            ssim2 = SSIM(target2.detach(), HR2.detach())
            ssim3 = SSIM(target3.detach(), HR3.detach())


            writer.add_scalar('Loss', loss.item(), epoch)

            writer.add_scalar('Metric/2x/PSNR', psnr1, epoch)
            writer.add_scalar('Metric/2x/SSIM', ssim1, epoch)

            writer.add_scalar('Metric/4x/PSNR', psnr2, epoch)
            writer.add_scalar('Metric/4x/SSIM', ssim2, epoch)

            writer.add_scalar('Metric/8x/PSNR', psnr3, epoch)
            writer.add_scalar('Metric/8x/SSIM', ssim3, epoch)
        
        if epoch!=0 and epoch%save_every==0:
            torch.save(model.state_dict(),os.path.join('Logs', FILENAME, f'Epoch_{epoch}.pth'))

    writer.close()

def test(num_workers, path, device = 'cpu'):

    testfiles = os.listdir('testdata')
    
    for testfile in testfiles:
        
        writer = SummaryWriter(log_dir=os.path.join('Logs',FILENAME,'Test',testfile))
        Dataset = Image_Dataset(train = False, filename=testfile)
        Dataloader = torch.utils.data.DataLoader(Dataset, 1, False, num_workers=num_workers)

        model = LapSRN()
        model.load_state_dict(torch.load(path))
        model.eval()


        with torch.no_grad():
            for idx, (input, target1, target2, target3) in enumerate(Dataloader):

                input = input.to(device)
                target1 = target1.to(device)
                target2 = target2.to(device)
                target3 = target3.to(device)


                print(idx)

                HR1,HR2,HR3 = model(input)
                loss = F.mse_loss(HR1, target1) + F.mse_loss(HR2, target2) + F.mse_loss(HR3, target3)


                psnr1 = PSNR(target1.detach(), HR1.detach())
                psnr2 = PSNR(target2.detach(), HR2.detach())
                psnr3 = PSNR(target3.detach(), HR3.detach())

                ssim1 = SSIM(target1.detach(), HR1.detach())
                ssim2 = SSIM(target2.detach(), HR2.detach())
                ssim3 = SSIM(target3.detach(), HR3.detach())


                writer.add_scalar('Loss', loss.item())

                writer.add_scalar('Metric/2x/PSNR', psnr1)
                writer.add_scalar('Metric/2x/SSIM', ssim1)

                writer.add_scalar('Metric/4x/PSNR', psnr2)
                writer.add_scalar('Metric/4x/SSIM', ssim2)

                writer.add_scalar('Metric/8x/PSNR', psnr3)
                writer.add_scalar('Metric/8x/SSIM', ssim3)

        writer.close()


if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument('--epochs',default=100,type=int, required=False)
    p.add_argument('--workers',default=80,type=int, required=False)
    p.add_argument('--batch_size',default=256,type=int, required=False)
    p.add_argument('--lr',default=1e-3,type=float, required=False)
    p.add_argument('--device',default='cpu',type=str, required=False)
    p.add_argument('--checkpoint',default=None,type=str, required=False)
    p.add_argument('--save_every',default=10,type=int, required=False)
    
    args = p.parse_args()



    FILENAME = f'LapSRN_{args.scale_factor}x_BS_{args.batch_size}_workers_{args.workers}_epochs_{args.epochs}_lr{args.lr}_{np.random.random()}'
    os.mkdir(os.path.join('Logs', FILENAME))

    if args.checkpoint is None:
        train(
            args.batch_size,
            args.workers,
            args.epochs,
            args.lr,
            args.device,
            args.save_every
        )
    else:
        test(
            args.workers,
            args.checkpoint,
            args.device
        )
