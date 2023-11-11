import random
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import torch
from model import ViT
from torch import nn

#### Hyperparameters
BATCH_SIZE = 512        #int, number of samples we want to pass into the training loop at each iteration.

IMG_SIZE = 224          #int, Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
PATCH_SIZE = 32         #int, Size of patches. image_size must be divisible by patch_size. The number of patches is:  n = (image_size // patch_size) ** 2 and n must be greater than 16.
NUM_CLASSES = 4         #int, Number of classes to classify.
DIM = 1024              #int, Last dimension of output tensor after linear transformation nn.Linear(..., dim).
DEPTH = 6               #int, Number of Transformer blocks.
NUM_HEADS = 16          #int, Number of heads in Multi-head Attention layer.
MLP_DIM = 2048          #int, Dimension of the MLP (FeedForward) layer, or hidden dim.
POOL = 'cls'            #string, either cls token pooling or mean pooling.  DEFAULT: 'cls'
CHANNELS = 3            #int, Number of image's channels. DEFAULT: 3
DIM_HEAD = 64           #DEFAULT: 64
DROPOUT = 0.001         #float between [0, 1], Dropout rate. DEFAULT: 0.
EMB_DROPOUT = 0.001     #float between [0, 1], Embedding dropout rate. DEFAULT: 0.

EPOCHS = 5
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION="gelu"




random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)




# Setup train and testing paths
#data_path = Path("data/")
#image_path = data_path / "Dataset"

image_path = Path("Dataset")

train_dir = image_path / "train"
val_dir = image_path / "val"
test_dir = image_path / "test"




train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), #change image size
    transforms.TrivialAugmentWide(num_magnitude_bins=31), # how intense
    #also make data augmentation if dataset is not large (?)
    # Flip the images randomly on the horizontal
    #transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    transforms.ToTensor() # use ToTensor() last to get everything between 0 & 1
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# Don't need to perform augmentation on the test data
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])



# 1. Load and transform data
train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(root=val_dir, transform=val_transforms)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)

# 2. Turn data into DataLoaders


#NUM_WORKERS = 4 # number cpu cores
#NUM_WORKERS = os.cpu_count()
#print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

# Create DataLoader's
train_dataloader = DataLoader(train_data,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True)
                                     #num_workers=NUM_WORKERS)

val_dataloader = DataLoader(test_data,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False)
                                    #num_workers=NUM_WORKERS)

test_dataloader = DataLoader(test_data,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False)
                                    #num_workers=NUM_WORKERS)



model = ViT(image_size=IMG_SIZE, patch_size=PATCH_SIZE, num_classes=NUM_CLASSES, dim=DIM, depth=DEPTH, heads=NUM_HEADS, mlp_dim=MLP_DIM, pool=POOL, channels=CHANNELS, dim_head=DIM_HEAD, dropout=DROPOUT, emb_dropout=EMB_DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)


for epoch in range(EPOCHS):
    # Training
    model.train()
    total_train_loss = 0.0
    correct_train = 0
    total_train = 0

    # Use tqdm to create a progress bar for the training loop
    with tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch') as train_bar:
        for i, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            avg_train_loss = total_train_loss / (i+1)
            train_accuracy = correct_train / total_train * 100

            train_bar.set_postfix({'Loss': avg_train_loss, 'Accuracy': train_accuracy})

    # Validation
    model.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = correct / total * 100

        print(f'Epoch [{epoch+1}/{EPOCHS}], Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}%')


# Save the trained model
torch.save(model.state_dict(), 'best.pth')
print('Trained model saved successfully.')



# Testing
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate test accuracy
test_accuracy = (sum([1 for p, l in zip(all_predictions, all_labels) if p == l]) / len(all_predictions)) * 100
print(f'Test Accuracy: {test_accuracy}%')
