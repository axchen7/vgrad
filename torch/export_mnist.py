import torch
from torchvision import datasets, transforms
from export_vgtensor import export_vgtensor

N_TRAIN = 10000
N_TEST = 500

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
export_vgtensor(train_images, "train_images.vgtensor")
export_vgtensor(test_images, "test_images.vgtensor")

# Export the training and test labels
export_vgtensor(train_labels, "train_labels.vgtensor")
export_vgtensor(test_labels, "test_labels.vgtensor")
