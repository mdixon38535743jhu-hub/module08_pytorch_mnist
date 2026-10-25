import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# LeNet CNN for MNIST classification
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Load MNIST training and testing datasets
def load_data(batch_size):
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Train the model and track loss and accuracy
def train_model(model, train_loader, test_loader, num_epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_list, acc_train, acc_test = [], [], []

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        acc_train.append(100 * correct / total)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        acc_test.append(100 * correct / total)
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