from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Initial block
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)  # 28x28x10
        self.bn1 = nn.BatchNorm2d(10)
        self.dropout1 = nn.Dropout(0.05)
        
        # Block 2
        self.conv2 = nn.Conv2d(10, 16, 3, padding=1)  # 28x28x16
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(0.05)
        self.pool1 = nn.MaxPool2d(2, 2)  # 14x14x16
        
        # Block 3
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)  # 14x14x16
        self.bn3 = nn.BatchNorm2d(16)
        self.dropout3 = nn.Dropout(0.1)
        
        # Block 4
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)  # 14x14x16
        self.bn4 = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout(0.1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 7x7x16
        
        # Final classification block
        self.conv5 = nn.Conv2d(16, 10, 1)  # 7x7x10
        self.gap = nn.AdaptiveAvgPool2d(1)  # 1x1x10

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool1(x)
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_description(desc=f'loss={loss.item():.4f} batch_id={batch_idx}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    torch.manual_seed(1)
    batch_size = 128
    epochs = 20
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data augmentation and normalization
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                      transform=train_transforms),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=test_transforms),
        batch_size=batch_size, shuffle=False, **kwargs)

    model = Net().to(device)
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Parameters: {total_params}')
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.1,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=10,
        final_div_factor=100
    )

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}')
        train(model, device, train_loader, optimizer, scheduler, epoch)
        accuracy = test(model, device, test_loader)
        if accuracy >= 99.4:
            print(f"Achieved 99.4% accuracy at epoch {epoch}. Stopping training.")
            break

if __name__ == '__main__':
    main() 