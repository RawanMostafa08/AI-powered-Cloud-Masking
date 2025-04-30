# Standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader

# Image I/O and processing
import rasterio
import pandas as pd

# Data augmentation and transforms
import albumentations as A
from albumentations import (
   Resize, Compose
)
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

# Model and segmentation
import segmentation_models_pytorch as smp

# Utilities
from tqdm import tqdm
from rle_encoder_decoder import rle_encode


class CloudSegmentationDataset(Dataset):
    def __init__(self, image_dir, image_files, transform=None):
        self.image_dir = image_dir
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load 4-band image: [4, H, W] → [H, W, 4]
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
            image = np.transpose(image, (1, 2, 0))  # [H, W, 4]

        # Check if the image has 4 channels
        if image.shape[2] == 4:
            # Select the first and fourth channels (index 0 and 3)
            image = image[..., [0, 3]]  # [H, W, 2]
            
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, img_name
    

test_image_dir = './test/test/data'
val_test_transform = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])
test_images = os.listdir(test_image_dir)

test_dataset = CloudSegmentationDataset(image_dir=test_image_dir, image_files=test_images, transform=val_test_transform)

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print(f"test set size: {len(test_loader.dataset)}")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {device}")
# Load the trained model
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",       
    encoder_weights="imagenet", 
    in_channels=2,                 
    classes=1,                     
)
model.load_state_dict(torch.load("../deeplabv3_final_model/deeplabv3_ckpt_epoch_18.pth", map_location=device))
model.to(device)

# Read the submission CSV
df = pd.read_csv('./sample_submission.csv', dtype={'id': str})

# Initialize lists for RLE encodings
predictions_rle = []


model.eval()

# Process each row in the DataFrame
with torch.no_grad():
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating on test set"):
        image, img_name = test_loader.dataset[idx]
        
        # Ensure image is in [2, H, W] format
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()  # [H, W, 2]
            image = image.permute(2, 0, 1)  # [2, H, W]
        
        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(device)  # [1, 2, H, W]
        
        # Run model
        output = model(image)
        output = torch.sigmoid(output).cpu().numpy()  # [1, 1, H, W]
        pred = output[0, 0]  # [H, W]
        
        # Binarize prediction
        pred_binary = (pred > 0.5).astype(np.uint8)
        
        # Compute RLE encoding
        rle = rle_encode(pred_binary)
        predictions_rle.append({"id": img_name, "segmentation": rle_encode(pred_binary)})
        

# Update the segmentation column in the DataFrame
df['segmentation'] = [pred['segmentation'] for pred in predictions_rle]

# Verify id column dtype before writing
print(f"id column dtype before writing: {df['id'].dtype}")
print(f"Sample IDs before writing: {df['id'].head(10).tolist()}")

# Save the updated DataFrame to CSV
df.to_csv('./team_13.csv', index=False)
print("Updated sample_submission.csv with new segmentations")