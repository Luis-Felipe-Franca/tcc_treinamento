import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torchvision import transforms
from tqdm import tqdm



# -----------------------
# 1. Dataset Definition
# -----------------------
class CrackSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # Sort so images & masks line up
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask

        image = np.array(image)
        mask = np.array(mask)
        mask = (mask > 128).astype(np.float32)  # convert to binary [0,1]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # Convert to torch tensors
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask

# -----------------------
# 2. Paths
# -----------------------
root = r"crack_segmentation_dataset"
train_images = os.path.join(root, "train", "images")
train_masks = os.path.join(root, "train", "masks")
test_images = os.path.join(root, "test", "images")
test_masks = os.path.join(root, "test", "masks")

# -----------------------
# 3. Datasets & Loaders
# -----------------------
train_dataset = CrackSegmentationDataset(train_images, train_masks)
test_dataset = CrackSegmentationDataset(test_images, test_masks)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# -----------------------
# 4. Model Definition
# -----------------------
model = smp.Unet(
    encoder_name="resnet34",        # backbone
    encoder_weights="imagenet",     # pretrained weights
    in_channels=3,                  # RGB
    classes=1,                      # Binary segmentation
)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# -----------------------
# 5. Loss & Optimizer
# -----------------------
bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def combined_loss(pred, target):
    return bce(pred, target) + dice_loss(pred, target)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -----------------------
# 6. Training Loop
# -----------------------
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f}")

# -----------------------
# 7. Save Model
# -----------------------
torch.save(model.state_dict(), "unet_crack_segmentation.pth")
print("✅ Training finished and model saved!")
