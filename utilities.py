import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Convolutional Neural Network for MNIST classification
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return self.fc2(x)


# Load MNIST training and testing datasets
def load_data(batch_size):
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Evaluate model accuracy on a dataset
def track_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


# Train model and track loss and accuracy
def train_model(model, train_loader, test_loader, num_epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_list, acc_train, acc_test = [], [], []

    for epoch in range(num_epochs):
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        acc_train.append(track_accuracy(model, train_loader, device))
        acc_test.append(track_accuracy(model, test_loader, device))
        print(f'Epoch {epoch + 1}/{num_epochs}: Train Accuracy = {acc_train[-1]:.2f}%, Test Accuracy = {acc_test[-1]:.2f}%')

    return loss_list, acc_train, acc_test


# Plot training loss and accuracy
def plot(loss_list, acc_train, acc_test, save_path='./figure.png'):
    plt.figure(figsize=(10, 4))

    # Training loss over iterations
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    # Training and testing accuracy over epochs
    plt.subplot(1, 2, 2)
    epochs = range(1, len(acc_train) + 1)
    plt.plot(epochs, acc_train, label='Train')
    plt.plot(epochs, acc_test, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Classification Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()