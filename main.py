import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os


# Define a custom dataset class for your .tif images
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.file_list[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image


# Define data transformations (you may need to adjust these)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Create dataset instances for training, validation, and testing
train_dataset = CustomDataset(data_dir='train_data', transform=data_transforms)
val_dataset = CustomDataset(data_dir='val_data', transform=data_transforms)
test_dataset = CustomDataset(data_dir='test_data', transform=data_transforms)

# Create data loaders for batching and shuffling
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Example usage of data loaders:
for images in train_loader:
    # Process images here (e.g., pass them through your model)
    pass
