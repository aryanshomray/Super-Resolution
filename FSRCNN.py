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

class FSRCNN(nn.Module):
    def __init__(self, scale_factor = 2, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


class Image_Dataset(torch.utils.data.Dataset):

    def __init__(self, scale_factor, train = True, filename = None):

        self.train = train
        self.scale_factor = scale_factor

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
        LR = cv2.resize(HR, (width//self.scale_factor, height//self.scale_factor))
        
        HR = torch.tensor((HR)/255).permute([2,0,1]).float()
        LR = torch.tensor((LR)/255).permute([2,0,1]).float()

        return LR, HR



def train(scale_factor, batch_size, num_workers, epochs, lr = 1e-3, device = 'cpu'):
    #Tensorboard Summary Writer Initialization
    writer = SummaryWriter(log_dir=os.path.join('Logs',FILENAME))

    Dataset = Image_Dataset(scale_factor = scale_factor)
    Dataloader = torch.utils.data.DataLoader(Dataset, batch_size, False, num_workers=num_workers)

    model = FSRCNN(scale_factor=scale_factor).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr)

    writer.add_scalar('Scale_Factor', scale_factor)
    writer.add_scalar('Epochs', epochs)
    writer.add_scalar('Batch_Size', batch_size)
    writer.add_scalar('LR', lr)

    for epoch in range(epochs):
        for idx, (input, target) in enumerate(Dataloader):

            input = input.to(device)
            target = target.to(device)

            if not epoch and not idx:
                writer.add_graph(model, input)

            print(idx)

            optimizer.zero_grad()
            output = model(input)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

            psnr = PSNR(target.detach(), output.detach())
            ssim = SSIM(target.detach(), output.detach())

            writer.add_scalar('Loss', loss.item(), epoch)
            writer.add_scalar('Metric/PSNR', psnr, epoch)
            writer.add_scalar('METRIC/SSIM', ssim, epoch)

        torch.save(model.load_state_dict(),os.path.join('Logs', FILENAME, f'Epoch_{epoch}.pth'))

    writer.close()

def test(scale_factor, num_workers, path):

    testfiles = os.listdir('testdata')
    
    for testfile in testfiles:
        
        writer = SummaryWriter(log_dir=os.path.join('Logs',FILENAME,'Test',testfile))
        Dataset = Image_Dataset(scale_factor = scale_factor, train = False, filenae=testfile)
        Dataloader = torch.utils.data.DataLoader(Dataset, 1, False, num_workers=num_workers)

        model = FSRCNN(scale_factor=scale_factor)
        model.load_state_dict(torch.load(path))
        mode.eval()

        writer.add_scalar('Scale_Factor', scale_factor)

        with torch.no_grad():
            for idx, (input, target) in enumerate(Dataloader):

                input = input.to(device)
                target = target.to(device)


                output = model(input)
                loss = F.mse_loss(output, target)

                psnr = PSNR(target.detach(), output.detach())
                ssim = SSIM(target.detach(), output.detach())

                writer.add_scalar('Loss', loss.item())
                writer.add_scalar('Metric/PSNR', psnr)
                writer.add_scalar('METRIC/SSIM', ssim)

        writer.close()


if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument('--scale_factor',default=2,type=int, required=False)
    p.add_argument('--epochs',default=100,type=int, required=False)
    p.add_argument('--workers',default=80,type=int, required=False)
    p.add_argument('--batch_size',default=256,type=int, required=False)
    p.add_argument('--lr',default=1e-3,type=float, required=False)
    p.add_argument('--device',default='cpu',type=str, required=False)
    p.add_argument('--checkpoint',default=None,type=str, required=False)
    
    args = p.parse_args()

    FILENAME = f'FSRCNN_{scale_factor}x_BS_{batch_size}_workers_{num_workers}_epochs_{epochs}_lr{lr}_{np.random.random()}'
    os.mkdir(os.path.join('Logs', FILENAME))

    if args['checkpoint'] is None:
        train(
            args['scale_factor'],
            args['batch_size'],
            args['workers'],
            args['epochs'],
            args['lr'],
            args['device']
        )
    else:
        test(
            args['scale_factor'],
            args['workers'],
            args['epochs'],
            args['checkpoint']
        )