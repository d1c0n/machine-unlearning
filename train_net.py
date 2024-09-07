import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from routedConv2d import (
    RoutedConv2dWithUnlearning,
)
from torch.nn import functional as F
# Assuming RoutedConv2dWithUnlearning is imported from the previous artifact


class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = RoutedConv2dWithUnlearning(3, 32, 3, n_filters=16, padding=1)
        self.conv2 = RoutedConv2dWithUnlearning(32, 64, 3, n_filters=32, padding=1)
        self.conv3 = RoutedConv2dWithUnlearning(64, 64, 3, n_filters=32, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x, target=None):
        x = self.pool(F.relu(self.conv1(x, target)))
        x = self.pool(F.relu(self.conv2(x, target)))
        x = F.relu(self.conv3(x, target))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_cifar10(
    num_epochs=50, batch_size=128, learning_rate=0.001, lambda_balance=0.1
):
    # Data preparation
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR10Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with labels
            outputs = model(inputs, labels)
            task_loss = criterion(outputs, labels)

            # Compute load balancing loss
            load_balancing_loss = 0
            for module in model.modules():
                if isinstance(module, RoutedConv2dWithUnlearning):
                    load_balancing_loss += module.get_load_balancing_loss(
                        inputs.size(0)
                    )

            # Combine losses
            total_loss = task_loss + lambda_balance * load_balancing_loss

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            if i % 100 == 99:  # Print every 100 mini-batches
                print(
                    f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}"
                )
                running_loss = 0.0

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)  # Note: we don't pass labels during evaluation
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set: {100 * correct / total:.2f}%")

        # Reset filter usage statistics at the end of each epoch
        for module in model.modules():
            if isinstance(module, RoutedConv2dWithUnlearning):
                module.reset_filter_usage()

    print("Finished Training")
    return model


# Run the training
trained_model = train_cifar10(lambda_balance=0.0001)

# Save the model
torch.save(trained_model.state_dict(), "cifar10_routed_model.pth")
