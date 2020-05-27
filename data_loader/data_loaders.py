import random
from PIL import Image
from os.path import join
from os import listdir
from torchvision.transforms import *
import torch.utils.data as data
from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class dataloader(BaseDataLoader):

    def __init__(self, image_dirs, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        self.dataset = TrainDatasetFromFolder(image_dirs, scale_factor=2)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def calculate_valid_crop_size(crop_size, scale_factor):
    return crop_size - (crop_size % scale_factor)


def verify(image_filenames, crop_size):
    final_paths = []
    for path in tqdm(image_filenames):
        img = load_img(path)
        width, height = img.size
        if width < crop_size or height < crop_size:
            continue
        else:
            final_paths.append(path)
    return final_paths

class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, image_dirs, is_gray=False, crop_size=192, rotate=True, fliplr=True,
                 fliptb=True, scale_factor=2):
        super(TrainDatasetFromFolder, self).__init__()

        self.image_filenames = []
        for image_dir in image_dirs:
            self.image_filenames.extend(join(image_dir, x) for x in sorted(
                listdir(image_dir)) if is_image_file(x))
        self.is_gray = is_gray
        self.crop_size = crop_size
        self.rotate = rotate
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.scale_factor = scale_factor
        self.image_filenames = verify(self.image_filenames, self.crop_size)

    def __getitem__(self, index):

        try:
            # load image
            img = load_img(self.image_filenames[index])

            # determine valid HR image size with scale factor
            self.crop_size = calculate_valid_crop_size(
                self.crop_size, self.scale_factor)
            hr_img_w = self.crop_size
            hr_img_h = self.crop_size

            # determine LR image size
            lr_img_w = hr_img_w // self.scale_factor
            lr_img_h = hr_img_h // self.scale_factor

            # random crop
            transform = RandomCrop(self.crop_size)
            img = transform(img)

            # random rotation between [90, 180, 270] degrees
            if self.rotate:
                rv = random.randint(1, 3)
                img = img.rotate(90 * rv, expand=True)

            # random horizontal flip
            if self.fliplr:
                transform = RandomHorizontalFlip()
                img = transform(img)

            # random vertical flip
            if self.fliptb:
                if random.random() < 0.5:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)

            # only Y-channel is super-resolved
            if self.is_gray:
                img = img.convert('YCbCr')
                # img, _, _ = img.split()

            # hr_img HR image
            hr_transform = Compose([ToTensor()])
            hr_img = hr_transform(img)

            # lr_img LR image
            lr_transform = Compose(
                [Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor()])
            lr_img = lr_transform(img)

            # Bicubic interpolated image
            bc_transform = Compose([ToPILImage(), Resize(
                (hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
            bc_img = bc_transform(lr_img)

            return {
                'LR':lr_img,
                'HR':hr_img,
                'BC':bc_img
            }

        except:
            return None

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, is_gray=False, scale_factor=4):
        super(TestDatasetFromFolder, self).__init__()

        self.image_filenames = [join(image_dir, x) for x in sorted(
            listdir(image_dir)) if is_image_file(x)]
        self.is_gray = is_gray
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index])

        # original HR image size
        w = img.size[0]
        h = img.size[1]

        # determine valid HR image size with scale factor
        hr_img_w = calculate_valid_crop_size(w, self.scale_factor)
        hr_img_h = calculate_valid_crop_size(h, self.scale_factor)

        # determine lr_img LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # only Y-channel is super-resolved
        if self.is_gray:
            img = img.convert('YCbCr')
            # img, _, _ = lr_img.split()

        # hr_img HR image
        hr_transform = Compose(
            [Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        hr_img = hr_transform(img)

        # lr_img LR image
        lr_transform = Compose(
            [Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        lr_img = lr_transform(img)

        # Bicubic interpolated image
        bc_transform = Compose([ToPILImage(), Resize(
            (hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        bc_img = bc_transform(lr_img)

        return lr_img, hr_img, bc_img

    def __len__(self):
        return len(self.image_filenames)
