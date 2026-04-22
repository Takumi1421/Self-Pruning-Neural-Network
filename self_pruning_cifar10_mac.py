import multiprocessing
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


multiprocessing.set_start_method("spawn", force=True)
torch.set_float32_matmul_precision("high")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
num_workers = 0
use_channels_last = device.type == "cuda"

torch.set_num_threads(os.cpu_count() or 1)
torch.set_num_interop_threads(max(1, (os.cpu_count() or 1) // 2))

print(f"Using device: {device}")
print(f"Using DataLoader workers: {num_workers}")
print("Expected ballpark after training: accuracy ~70-80%, sparsity depends on lambda and is often 10-60%.")


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        self.temperature = 1.0
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores / self.temperature)
        return F.linear(x, self.weight * gates, self.bias)

    def sparsity_loss(self):
        return torch.sigmoid(self.gate_scores).mean()

    def gate_values(self):
        return torch.sigmoid(self.gate_scores).detach()

    def update_temperature(self, decay=0.95, min_temp=0.1):
        self.temperature = max(min_temp, self.temperature * decay)


class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc1 = PrunableLinear(64 * 8 * 8, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = PrunableLinear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = PrunableLinear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(F.relu(self.bn1(self.fc1(x)), inplace=True))
        x = self.dropout(F.relu(self.bn2(self.fc2(x)), inplace=True))
        x = F.relu(self.bn3(self.fc3(x)), inplace=True)
        return self.fc4(x)

    def prunable_layers(self):
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_regularization(self):
        return sum(layer.sparsity_loss() for layer in self.prunable_layers())

    def compute_sparsity(self, threshold=0.01):
        total = 0
        pruned = 0
        for layer in self.prunable_layers():
            gates = layer.gate_values()
            total += gates.numel()
            pruned += (gates < threshold).sum().item()
        return pruned / total if total else 0.0


def move_batch(images, labels):
    if use_channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    images = images.to(device)
    labels = labels.to(device)
    return images, labels


def get_cifar10_loaders(batch_size=128):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    loader_kwargs = {
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["multiprocessing_context"] = "spawn"

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, lambda_sparse):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = move_batch(images, labels)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        cls_loss = F.cross_entropy(logits, labels)
        sparse_loss = model.sparsity_regularization()
        loss = cls_loss + lambda_sparse * sparse_loss
        loss.backward()
        optimizer.step()

        for layer in model.prunable_layers():
            layer.update_temperature()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = move_batch(images, labels)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_model(lambda_sparse, num_epochs=10, batch_size=128, lr=1e-3):
    train_loader, test_loader = get_cifar10_loaders(batch_size)
    model = SelfPruningNet().to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    history = []

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, lambda_sparse)
        scheduler.step()
        acc = evaluate(model, test_loader)
        sparsity = model.compute_sparsity()
        epoch_time = time.time() - epoch_start
        history.append({
            "epoch": epoch,
            "loss": loss,
            "accuracy": acc,
            "sparsity": sparsity,
            "time_seconds": epoch_time,
        })
        print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"lambda={lambda_sparse} | "
            f"loss={loss:.4f} | "
            f"acc={acc * 100:.2f}% | "
            f"sparsity={sparsity * 100:.1f}% | "
            f"time={epoch_time:.1f}s"
        )

    final_accuracy = evaluate(model, test_loader)
    final_sparsity = model.compute_sparsity()
    return {
        "lambda": lambda_sparse,
        "accuracy": final_accuracy,
        "sparsity": final_sparsity,
        "history": history,
    }


if __name__ == "__main__":
    lambdas = [0.0, 1e-7, 1e-6, 1e-5, 5e-5]
    num_epochs = 10
    batch_size = 128

    results = []
    for lambda_sparse in lambdas:
        print(f"\nTraining with lambda = {lambda_sparse}")
        result = train_model(lambda_sparse=lambda_sparse, num_epochs=num_epochs, batch_size=batch_size)
        results.append(result)
        print(
            f"Final result | lambda={result['lambda']} | "
            f"accuracy={result['accuracy'] * 100:.2f}% | "
            f"sparsity={result['sparsity'] * 100:.1f}%"
        )

    best = max(results, key=lambda x: x["accuracy"])
    print("\nSummary")
    for result in results:
        print(
            f"lambda={result['lambda']} | "
            f"accuracy={result['accuracy'] * 100:.2f}% | "
            f"sparsity={result['sparsity'] * 100:.1f}%"
        )

    print(
        f"\nBest by accuracy | lambda={best['lambda']} | "
        f"accuracy={best['accuracy'] * 100:.2f}% | "
        f"sparsity={best['sparsity'] * 100:.1f}%"
    )
