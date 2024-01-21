from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        ABC_img = Image.open(A_path).convert('RGB')
        
        # Assuming the combined image is horizontally aligned (B, C, D)
        w, h = ABC_img.size
        w_third = int(w / 3)

        # Crop B, C, D images from the combined image
        B = ABC_img.crop((0, 0, w_third, h))
        C = ABC_img.crop((w_third, 0, w_third * 2, h))
        D = ABC_img.crop((w_third * 2, 0, w, h))

        B = self.transform(B)
        C = self.transform(C)
        D = self.transform(D)

        # Combine B, C, and D into one tensor (e.g., stacking)
        # Adjust this according to how you want to combine these images
        A = torch.cat([B, C, D], dim=0)  # Stacking along a new dimension

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
