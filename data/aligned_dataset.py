import os
import torch
import random
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # RandomResizedCrop-like behavior. This was the simplest way to use the same randomness throughout all transforms.
        scale_min, scale_max = 0.6, 1.0  # Scale range
        w, h = AB.size
        w4 = int(w / 4)

        scale = random.uniform(scale_min, scale_max)
        crop_width = int(w4 * scale)
        crop_height = int(h * scale)

        # Randomly choose the top-left corner of the cropping area for the entire batch
        left_margin = random.randint(0, w4 - crop_width)
        top_margin = random.randint(0, h - crop_height)

        # Apply the same crop to each quarter of the image
        A = AB.crop((left_margin, top_margin, left_margin + crop_width, top_margin + crop_height))
        B = AB.crop((w4 + left_margin, top_margin, w4 + left_margin + crop_width, top_margin + crop_height))
        C = AB.crop((w4*2 + left_margin, top_margin, w4*2 + left_margin + crop_width, top_margin + crop_height))
        D = AB.crop((w4*3 + left_margin, top_margin, w4*3 + left_margin + crop_width, top_margin + crop_height))

        transform_params = get_params(self.opt, A.size)
        flip = random.random() < 0.4
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), flip=flip)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), flip=flip)

        A = A_transform(A)
        B = B_transform(B)
        C = B_transform(C)
        D = B_transform(D)
        B = torch.cat((B, C, D))

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
