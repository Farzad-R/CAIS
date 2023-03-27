# https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
import torch 
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from pyprojroot import here
from PytorchCNN import CNN
from torch.autograd import Variable
from tqdm import tqdm

batch_size = 128
learning_rate = 0.001
epochs = 1
early_stopper_paitient = 2

# load data
mnist_train = dsets.MNIST(root=here("data/"),
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root=here("data/"),
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
print('Size of the training dataset is {}'.format(mnist_train.data.size()))
print('Size of the testing dataset'.format(mnist_test.data.size()))

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True)

# Instantiate CNN model
model = CNN()

# Move the model to GPU
if torch.cuda.device_count() > 0:
    print("Number of available GPUS:", torch.cuda.device_count())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Distribute the model over the available GPUs
# model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
model = torch.nn.DataParallel(model, device_ids=[0])


# Define the optimizer and loss function
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# find the number of training steps (number of batches)
training_steps = len(mnist_train) // batch_size


print('Total number of training steps is : {0:2.0f}'.format(training_steps))
#%%
print('Training...')
train_loss = []
train_accuracy = []
model.train()
for epoch in range(epochs):
    avg_loss = 0
    avg_acc = 0
    for i, (batch_X, batch_Y) in tqdm(enumerate(data_loader), total=training_steps):
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        X = Variable(batch_X)    # image is already size of (28x28), no reshape
        Y = Variable(batch_Y)    # label is not one-hot encoded

        optimizer.zero_grad() # <= initialization of the gradients
        
        # forward propagation
        hypothesis = model(X)
        loss = criterion(hypothesis, Y) # <= compute the loss function

        # Backward propagation
        loss.backward() # <= compute the gradient of the loss function   
        optimizer.step() # <= Update the gradients
             
        # Print some performance to monitor the training
        prediction = hypothesis.data.max(dim=1)[1]
        train_accuracy.append(((prediction.data == Y.data).float().mean()).item())
        train_loss.append(loss.item())

        avg_acc +=  train_accuracy[-1] / training_steps
        avg_loss += loss.data / training_steps

    print("Epoch:", epoch + 1, "| averag loss:", avg_loss.item(), "avg acc:", avg_acc)
#%%
# Evaluate the model
model.eval()

X_test = Variable(mnist_test.data.view(len(mnist_test), 1, 28, 28).float())
Y_test = Variable(mnist_test.targets)
prediction = model(X_test)

# Compute accuracy
correct_prediction = (torch.max(prediction.data, dim=1)[1] == Y_test.to(device))
accuracy = correct_prediction.float().mean().item()
print('Test Accuracy: {:2.2f} %'.format(accuracy*100))

