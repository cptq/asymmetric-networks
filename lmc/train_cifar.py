'''Train CIFAR10 with PyTorch.
Built off of https://github.com/kuangliu/pytorch-cifar/tree/master
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.mlp import ImageMLP
from utils import progress_bar


def main():
  parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
  parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
  parser.add_argument('--resume', '-r', action='store_true',
                      help='resume from checkpoint')
  args = parser.parse_args()


  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  best_val_loss = float('inf')  # best val loss
  start_epoch = 0  # start from epoch 0 or last checkpoint epoch

  # Data
  print('==> Preparing data..')
  transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  trainset = torchvision.datasets.CIFAR10(
      root='./data', train=True, download=True, transform=transform_train)
  train_num = int(len(trainset)*.9)
  trainset, valset = torch.utils.data.Subset(trainset, range(train_num)), torch.utils.data.Subset(trainset, range(train_num, len(trainset)))
  testset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=transform_test)
  
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
  valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

  classes = ('plane', 'car', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck')

  # Model
  print('==> Building model..')
  net = ImageMLP(32*32*3, 64, 10, 3)
  net = net.to(device)
  #if device == 'cuda':
      #net = torch.nn.DataParallel(net)

  if args.resume:
      # Load checkpoint.
      print('==> Resuming from checkpoint..')
      assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
      checkpoint = torch.load('./checkpoint/ckpt.pth')
      net.load_state_dict(checkpoint['net'])
      best_val_loss = checkpoint['best_val_loss']
      start_epoch = checkpoint['epoch']

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

  for epoch in range(start_epoch, start_epoch+200):
    
      train_loss, train_acc = train(net, epoch, trainloader, device, optimizer, criterion)
      val_loss, val_acc = test(net, epoch, valloader, device, criterion)
      test_loss, test_acc = test(net, epoch, testloader, device, criterion)
      
      # Save checkpoint.
      if val_loss < best_val_loss:
          print('Saving..')
          state = {
              'net': net.state_dict(),
              'best_val_loss': best_val_loss,
              'epoch': epoch,
          }
          if not os.path.isdir('checkpoint'):
              os.mkdir('checkpoint')
          torch.save(state, './checkpoint/ckpt.pth')
          best_val_loss = val_loss
      scheduler.step()

# Training
def train(net, epoch, loader, device, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss, correct / total


def test(net, epoch, loader, device, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    
    return test_loss, correct / total

if __name__ == '__main__':
  main()