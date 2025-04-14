import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils import data_split

class NumpyMNISTDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MLPClassifierTorch(nn.Module):
    def __init__(self):
        super(MLPClassifierTorch, self).__init__()

        hidden_layer_sizes = (72, 62, 62, 72, 23, 60, 36, 61, 54, 98, 41, 33, 62, 71, 53, 11)
        input_size = 784
        output_size = 10

        layer_sizes = [input_size] + list(hidden_layer_sizes) + [output_size]
        layers = []

        for in_size, out_size in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch}: Loss = {total_loss:.4f}, Accuracy = {acc:.2f}%")

def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct / total * 100:.2f}%")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X_train, X_test, y_train, y_test = data_split()

    train_dataset = NumpyMNISTDataset(X_train, y_train)
    test_dataset  = NumpyMNISTDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=1000)

    model = MLPClassifierTorch().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)

    test(model, device, test_loader)

    torch.save(model.state_dict(), "mlp_mnist_model.pth")
    print("Model saved as 'mlp_mnist_model.pth'")


if __name__ == "__main__":
    main()