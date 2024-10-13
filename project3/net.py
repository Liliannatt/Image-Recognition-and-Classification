import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(96 * 2 * 2, 512)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.conv3(x))
        x = x.view(-1, 96 * 2 * 2)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        return x


class Net_A1(nn.Module):
    def __init__(self):
        super(Net_A1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.conv3(x))
        x = x.view(-1, 256 * 2 * 2)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        return x

class Net_A2(nn.Module):
    def __init__(self):
        super(Net_A2, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(96 * 2 * 2, 512)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.relu_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.conv3(x))
        x = x.view(-1, 96 * 2 * 2)
        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_B(nn.Module):
    def __init__(self):
        super(Net_B, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256 * 1 * 1, 512)  # 需要根据输出大小调整
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.conv3(x))
        x = x.view(-1, 256 * 1 * 1)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        return x


class Net_C(nn.Module):
    def __init__(self):
        super(Net_C, self).__init__()
        negative_slope = 0.01
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=1, padding=0)
        self.lrelu1 = nn.LeakyReLU(negative_slope)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=0)
        self.lrelu2 = nn.LeakyReLU(negative_slope)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(96 * 2 * 2, 512)
        self.lrelu_fc1 = nn.LeakyReLU(negative_slope)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.lrelu1(self.conv1(x)))
        x = self.pool2(self.lrelu2(self.conv2(x)))
        x = self.pool3(self.conv3(x))
        x = x.view(-1, 96 * 2 * 2)
        x = self.lrelu_fc1(self.fc1(x))
        x = self.fc2(x)
        return x


class Net_D(nn.Module):
    def __init__(self):
        super(Net_D, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(96 * 2 * 2, 512)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.conv3(x))
        x = x.view(-1, 96 * 2 * 2)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Net_E(nn.Module):
    def __init__(self):
        super(Net_E, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(24)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(48)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(96)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(96 * 2 * 2, 512)
        self.relu_fc1 = nn.ReLU()
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.bn1(self.relu1(self.conv1(x))))
        x = self.pool2(self.bn2(self.relu2(self.conv2(x))))
        x = self.pool3(self.bn3(self.conv3(x)))
        x = x.view(-1, 96 * 2 * 2)
        x = self.bn_fc(self.relu_fc1(self.fc1(x)))
        x = self.fc2(x)
        return x

class Net_5A(nn.Module):
    def __init__(self):
        super(Net_5A, self).__init__()
        negative_slope = 0.01
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)
        self.lrelu1 = nn.LeakyReLU(negative_slope)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.lrelu2 = nn.LeakyReLU(negative_slope)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.lrelu_fc1 = nn.LeakyReLU(negative_slope)
        self.bn_fc = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.lrelu_fc2 = nn.LeakyReLU(negative_slope)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.bn1(self.lrelu1(self.conv1(x))))
        x = self.pool2(self.bn2(self.lrelu2(self.conv2(x))))
        x = self.pool3(self.bn3(self.conv3(x)))
        x = x.view(-1, 256 * 2 * 2)
        x = self.bn_fc(self.lrelu_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.lrelu_fc2(self.fc2(x))
        x = self.fc3(x)
        return x
