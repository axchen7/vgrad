import torch
from torchvision import datasets, transforms
from export_vgtensor import export_vgtensor
import os

N_TRAIN = 10000
N_TEST = 500

DATA_DIR = "../vgrad/examples/data"
os.makedirs(DATA_DIR, exist_ok=True)

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the training data
trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=N_TRAIN, shuffle=True)

# Download and load the test data
testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=N_TEST, shuffle=True)

# Get one batch of training images and labels
train_images, train_labels = next(iter(trainloader))

# Get one batch of test images and labels
test_images, test_labels = next(iter(testloader))

# Remove the batch dimension
train_images = train_images.squeeze(1)
test_images = test_images.squeeze(1)

# Convert labels to C++ int
train_labels = train_labels.int()
test_labels = test_labels.int()

print(f"Train images tensor shape: {train_images.shape}, dtype: {train_images.dtype}")
print(f"Test images tensor shape: {test_images.shape}, dtype: {test_images.dtype}")
print(f"Train labels tensor shape: {train_labels.shape}, dtype: {train_labels.dtype}")
print(f"Test labels tensor shape: {test_labels.shape}, dtype: {test_labels.dtype}")

# Export the training and test images
export_vgtensor(train_images, os.path.join(DATA_DIR, "train_images.vgtensor"))
export_vgtensor(test_images, os.path.join(DATA_DIR, "test_images.vgtensor"))

# Export the training and test labels
export_vgtensor(train_labels, os.path.join(DATA_DIR, "train_labels.vgtensor"))
export_vgtensor(test_labels, os.path.join(DATA_DIR, "test_labels.vgtensor"))

# ===== model and training =====


class Model(torch.nn.Module):
    def __init__(self, in_size, out_size, inner_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_size, inner_size)
        self.layer2 = torch.nn.Linear(inner_size, out_size)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


inner = 16

img_size = 28
flat_size = img_size * img_size

classes = 10

train_labels = train_labels.long()
test_labels = test_labels.long()

train_flat = train_images.view(-1, flat_size)
test_flat = test_images.view(-1, flat_size)

model = Model(flat_size, classes, inner)

lr = 0.1
epochs = 400

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    train_out = model(train_flat)
    train_loss = torch.nn.functional.cross_entropy(train_out, train_labels)
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    test_out = model(test_flat)
    test_loss = torch.nn.functional.cross_entropy(test_out, test_labels)

    test_acc = (test_out.argmax(dim=1) == test_labels).float().mean()

    print(
        f"Epoch: {epoch}\ttrain loss: {train_loss.item():.5f}\ttest loss: {test_loss.item():.5f}\ttest acc: {test_acc.item():.5f}"
    )
