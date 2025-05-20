"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""

import os
import random
import torch
import pandas as pd
import numpy as np
import hashlib
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import torch.nn.functional as F
import torchvision.transforms.functional as TF

NUM_WORKERS = os.cpu_count()

class MeanStdNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

class MinMaxNormalize:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x):
        return (x - self.min_val) / (self.max_val - self.min_val + 1e-8)

import torch
import random
import torchvision.transforms.functional as TF

class HyperspectralAugmentation:
    def __init__(self, level='low', mode='train', crop_size=128):
        assert level in ['null', 'low', 'medium', 'high'], "level must be 'low', 'medium', or 'high'"
        assert mode in ['train', 'validation']

        self.level = level
        self.mode = mode
        self.crop_size = crop_size
        
        # Define parameters per level
        if level == 'null':
            self.rotate_angles = []  # No rotation
            self.flip = False
            self.spectral_dropout_prob = 0.0
            self.noise_std = 0.0
            self.mask_rect_prob = 0.0
            self.mask_circle_prob = 0.0

        if level == 'low':
            self.rotate_angles = []  # No rotation
            self.flip = True
            self.spectral_dropout_prob = 0.05
            self.noise_std = 0.002
            self.mask_rect_prob = 0.1
            self.mask_circle_prob = 0.1

        elif level == 'medium':
            self.rotate_angles = [0, 90, 180, 270]
            self.flip = True
            self.spectral_dropout_prob = 0.1
            self.noise_std = 0.005
            self.mask_rect_prob = 0.2
            self.mask_circle_prob = 0.2

        elif level == 'high':
            self.rotate_angles = [0, 45, 90, 135, 180, 225, 270, 315]
            self.flip = True
            self.spectral_dropout_prob = 0.2
            self.noise_std = 0.01
            self.mask_rect_prob = 0.4
            self.mask_circle_prob = 0.4
    
    def _get_image_seed(self, img: torch.Tensor) -> int:
        
        # Ensure tensor is on CPU and detached from the computation graph
        img = img.cpu().detach()

        # Calculate mean, std, min, and max of the image tensor
        mean_val = img.mean().item()
        std_val = img.std().item()
        min_val = img.min().item()
        max_val = img.max().item()

        # Create a string from the values and encode it
        hash_input = f"{mean_val:.6f}_{std_val:.6f}_{min_val:.6f}_{max_val:.6f}".encode()

        # Hash the string using SHA256
        hash_val = hashlib.sha256(hash_input).hexdigest()

        # Convert the hash value to an integer (first 8 hex digits)
        seed = int(hash_val[:8], 16)

        return seed

    def __call__(self, img):

        """
        img: Tensor [C, H, W]
        """
        C, H, W = img.shape
    
        if self.mode == 'validation':
            seed = self._get_image_seed(img)
            rnd = random.Random(seed)
        else:
            rnd = random

        # Random crop to (125, X, X) - Training mode only
        if H > self.crop_size and W > self.crop_size:
            if self.mode == 'train':
                top = rnd.randint(0, H - self.crop_size)
                left = rnd.randint(0, W - self.crop_size)
                img = img[:, top:top + self.crop_size, left:left + self.crop_size]
            elif self.mode == 'validation':
                # Center crop to keep deterministic size
                top = (H - self.crop_size) // 2
                left = (W - self.crop_size) // 2
                img = img[:, top:top + self.crop_size, left:left + self.crop_size]
                H, W = self.crop_size, self.crop_size

        # Convert to [H, W, C] for rotation
        img = img.permute(1, 2, 0)

        # Rotation
        if self.rotate_angles:
            angle = rnd.choice(self.rotate_angles)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)

        img = img.permute(2, 0, 1)

        # Flip
        if self.flip:
            if rnd.random() > 0.5:
                img = torch.flip(img, dims=[1])  # Vertical
            if rnd.random() > 0.5:
                img = torch.flip(img, dims=[2])  # Horizontal

        # Skip all remaining augmentations in validation
        if self.mode == 'validation':
            return img

        # Training-only augmentations
        if self.spectral_dropout_prob > 0:
            num_drop = int(C * self.spectral_dropout_prob)
            drop_indices = rnd.sample(range(C), num_drop)
            img[drop_indices, :, :] = 0.0

        if self.noise_std > 0:
            noise = torch.randn_like(img) * self.noise_std
            img = img + noise

        if rnd.random() < self.mask_rect_prob:
            rect_h = rnd.randint(H // 10, H // 3)
            rect_w = rnd.randint(W // 10, W // 3)
            top = rnd.randint(0, H - rect_h)
            left = rnd.randint(0, W - rect_w)
            img[:, top:top + rect_h, left:left + rect_w] = 0.0

        if rnd.random() < self.mask_circle_prob:
            radius = rnd.randint(min(H, W) // 10, min(H, W) // 4)
            center_x = rnd.randint(radius, W - radius)
            center_y = rnd.randint(radius, H - radius)
            yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            mask = ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= radius ** 2
            img[:, mask] = 0.0

        return img


class HyperspectralDataset(Dataset):
    def __init__(
        self,
        data_dir,
        labels_file,
        transform=None,
        output_type='reg',
        num_classes=100,
        ):
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(labels_file)
        self.transform = transform
        self.output_type = output_type if output_type in ['reg', 'ordreg'] else None
        self.num_classes = num_classes

    def __len__(self):
        return len(self.labels_df)
    
    #@staticmethod
    # CORN-style ordinal classification:
    def encode_ordinal_target(self, label):
        # label in range 1–100 → target vector of length 99
        return torch.FloatTensor([1 if i < label - 1 else 0 for i in range(self.num_classes - 1)])

    @staticmethod
    def decode_ordinal_prediction(logits):
        # logits → sigmoid → binary decisions
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1) + 1  # Convert to label in 1–100

    def __getitem__(self, idx):
        sample_id = self.labels_df.iloc[idx]['id']
        label = float(self.labels_df.iloc[idx]['label'])
        
        if self.output_type == "reg":            
            label = (label - 1.0) / (self.num_classes - 1)
        elif self.output_type == 'ordreg':
            label = int((label - 1) // (100 // self.num_classes)) + 1
            label = self.encode_ordinal_target(label)
            #label = self.encode_ordinal_target(label, num_classes=100)

        npy_path = os.path.join(self.data_dir, sample_id)

        try:
            image = np.load(npy_path)
        
            # Convert to float32 and scale if needed (e.g., normalization)
            image = image.astype(np.float32)

            # Convert to tensor and permute to (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1)  # Shape: (125, 128, 128)

            # Resize spatial dimensions to (128, 128)
            image = F.interpolate(image.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False)
            image = image.squeeze(0)  # (C, 128, 128)

            # Handle tiling
            tile_n = self.labels_df.iloc[idx].get('tile', None)
            if tile_n is not None:
                if tile_n == 0:
                    image = image[:, :64, :64]
                elif tile_n == 1:
                    image = image[:, :64, 64:]
                elif tile_n == 2:
                    image = image[:, 64:, :64]
                else:
                    image = image[:, 64:, 64:]
                
            if self.transform:           
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.float32)

        except Exception as e:
            print(f"Warning: Skipping file '{npy_path}' at index {idx}. Error: {e}")
            next_idx = (idx + 1) % len(self.labels_df)
            return self.__getitem__(next_idx)

    
def create_dataloaders(
        train_dir: str, 
        test_dir: str, 
        train_labels: str,
        test_labels: str,
        train_transform, 
        test_transform,
        batch_size: int, 
        num_workers: int=4,        
        output_type: str='reg',
        num_classes: int=100
    ):
    train_data = HyperspectralDataset(train_dir, train_labels, transform=train_transform, output_type=output_type, num_classes=num_classes)
    test_data = HyperspectralDataset(test_dir, test_labels, transform=test_transform, output_type=output_type, num_classes=num_classes)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader