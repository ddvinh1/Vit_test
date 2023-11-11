import model
from PIL import Image, ExifTags
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from pathlib import Path
import torch
from model import ViT
from torchvision import transforms
import numpy as np

# Replace 'path/to/your/image.jpg' with the actual path to your image
image_path = r"C:\Users\Vinh\Downloads\Test ViT-20231110T041825Z-001\Test ViT\20231110_072038.jpg"

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

EPOCHS = 20
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION="gelu"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViT(image_size=IMG_SIZE, patch_size=PATCH_SIZE, num_classes=NUM_CLASSES, dim=DIM, depth=DEPTH, heads=NUM_HEADS, mlp_dim=MLP_DIM, pool=POOL, channels=CHANNELS, dim_head=DIM_HEAD, dropout=DROPOUT, emb_dropout=EMB_DROPOUT).to(device)
model.load_state_dict(torch.load('best.pth'))
model.eval()


# Step 3: Preprocess the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open(image_path).convert("RGB")
image = image.rotate(0, expand=True)
input_tensor = transform(image).unsqueeze(0).to(device)

# Step 4: Make predictions
with torch.no_grad():
    model.eval()
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)
    _, predicted_class = torch.max(output, 1)



# Step 5: Display the results
class_names= sorted([entry.name for entry in list(os.scandir(Path("Dataset/train/")))])
class_labels = class_names  # Replace with your actual class labels
predicted_label = class_labels[predicted_class.item()]
confidence = torch.max(probabilities).item() * 100

# Plot the input image
plt.imshow(np.array(image))
plt.title(f'Predicted Class: {predicted_label}\nConfidence: {confidence:.2f}%')
plt.axis('off')
plt.show()
