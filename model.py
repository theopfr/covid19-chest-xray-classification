
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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

    def _visualize_features(self, feature_maps, dim: tuple=(), title: str=""):
        try:
            x, y = dim
            fig, axs = plt.subplots(x, y)
            c = 0
            for i in range(x):
                for j in range(y):
                    axs[i][j].matshow(feature_maps.detach().cpu().numpy()[0][c])
                    c += 1

            fig.suptitle(title)
            plt.show()

        except Exception as e:
            print(e)

    def forward(self, x, print_: bool=False, visualize: bool=False):
        if print_: print(x.size())

        x = self.block1(x)        
        if print_: print(x.size())
        if visualize: self._visualize_features(x, dim=(5, 5))

        x = self.block2(x)
        if print_: print(x.size())
        x = self.maxpool3x3(x)
        if visualize: self._visualize_features(x, dim=(6, 6))

        x = self.block3(x)
        if print_: print(x.size())
        if visualize: self._visualize_features(x, dim=(6, 6))

        x = self.block4(x)
        if print_: print(x.size())
        x = self.maxpool3x3(x)
        if visualize: self._visualize_features(x, dim=(6, 6))

        x = self.block5(x)
        if print_: print(x.size())
        if visualize: self._visualize_features(x, dim=(6, 6))

        x = self.block6(x)
        if print_: print(x.size())
        x = self.maxpool3x3(x)
        if visualize: self._visualize_features(x, dim=(6, 6))

        x = self.block7(x)
        if print_: print(x.size())
        if visualize: self._visualize_features(x, dim=(6, 6))

        x = self.block8(x)
        if print_: print(x.size())
        x = self.maxpool3x3(x)
        if visualize: self._visualize_features(x, dim=(6, 6))

        x = self.block9(x)
        if print_: print(x.size())

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

