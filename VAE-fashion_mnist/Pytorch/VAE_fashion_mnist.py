import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.datasets as datasets  
from torch.utils.data import DataLoader  
from Model import VariationalAutoEncoder
from pyprojroot import here
from functions import train, inference

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 784
Z_DIM = 20
H_DIM = 200
NUM_EPOCHS = 20
BATCH_SIZE = 32
LR_RATE = 3e-4
digit = 3

# Loading MNIST  dataset 
dataset = datasets.MNIST(root=here("data/MNIST/raw"), train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
# Print number of batches
print("Number of batches", len(train_loader))


# Initialize model, optimizer, loss
model = VariationalAutoEncoder(INPUT_DIM, Z_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")

# Run training
train(NUM_EPOCHS, model, optimizer, loss_fn, train_loader, INPUT_DIM, device)
# Let's plot an example
model = model.to(torch.device('cpu'))

# generate an image and save it in the same directory
inference(digit, model, dataset, num_examples=1)


