
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, module):
        super(ResBlock, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x) + x


class Model(nn.Module):
    def __init__(self, dropout_chance=0.0):
        super(Model, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.block2 = ResBlock(
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(32)
            )
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.block4 = ResBlock(
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(64)
            )
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.block6 = ResBlock(
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(128)
            )
        )

        self.block7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.block8 = ResBlock(
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.BatchNorm2d(256)
            )
        )

        self.block9 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(256)
        )

        self.maxpool2x2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool3x3 = nn.MaxPool2d(kernel_size=(3, 3))

        self.dropout = nn.Dropout2d(p=dropout_chance)

        self.dense1 = nn.Linear(1024, 512)
        self.dense2 = nn.Linear(512, 128)
        self.dense3 = nn.Linear(128, 3)

    def forward(self, x, print_: bool=False, visualize: bool=False):
        if print_: print(x.size())

        x = self.block1(x)        
        if print_: print(x.size())

        x = self.block2(x)
        x = self.maxpool3x3(x)
        if print_: print(x.size())

        x = self.block3(x)
        if print_: print(x.size())

        x = self.block4(x)
        x = self.maxpool3x3(x)
        if print_: print(x.size())

        x = self.block5(x)
        if print_: print(x.size())

        x = self.block6(x)
        x = self.maxpool3x3(x)
        if print_: print(x.size())

        x = self.block7(x)
        if print_: print(x.size())

        x = self.block8(x)
        x = self.maxpool3x3(x)
        if print_: print(x.size())

        x = self.block9(x)
        if print_: print("u", x.size())

        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
        x = F.softmax(self.dense3(x), dim=1)

        return x


"""x = torch.Tensor(torch.rand((1, 1, 512, 512))).cuda()

model = Model().cuda()
x = model.forward(x, print_=True)"""

