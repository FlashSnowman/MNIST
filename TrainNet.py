from MyNet import MLPNet, CNNNet
import torch.nn as nn
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class Train:
    def __init__(self):
        # self.net = MLPNet().cuda()
        self.net = CNNNet().cuda()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.mean, self.std = self.mean_std()
        self.trainSet, self.testSet = self.data_set()

    def mean_std(self):
        train = datasets.MNIST(root="datasets/", train=True, download=False, transform=transforms.ToTensor())
        test = datasets.MNIST(root="datasets/", train=False, download=False, transform=transforms.ToTensor())
        sets = train + test
        loader = torch.utils.data.DataLoader(sets, batch_size=len(sets), shuffle=True)
        data = next(iter(loader))[0]
        mean = float(torch.mean(data, dim=(0, 2, 3)).numpy())
        std = float(torch.std(data, dim=(0, 2, 3)).numpy())
        return mean, std

    def data_set(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.mean,), (self.std,))
        ])
        trainSet = datasets.MNIST(root="datasets/", train=True, download=False, transform=transform)
        testSet = datasets.MNIST(root="datasets", train=False, download=False, transform=transform)
        return trainSet, testSet

    def data_loader(self):
        trainLoader = torch.utils.data.DataLoader(self.trainSet, batch_size=512, shuffle=True)
        testLoader = torch.utils.data.DataLoader(self.testSet, batch_size=512, shuffle=True)
        return trainLoader, testLoader

    def train(self):
        trainLoader, testLoader = self.data_loader()
        epochs = 10
        losses = []
        for i in range(epochs):
            for j, (x, y) in enumerate(trainLoader):
                x = x.cuda()
                # y = torch.zeros(y.size(0), epochs).scatter_(1, y.view(-1, 1), 1).cuda()
                y = nn.functional.one_hot(y.long()).float().cuda()
                out = self.net(x)
                loss = self.loss_func(out, y)
                if j % 10 == 0:
                    print("[epochs: {0} - {1}/{2}]loss: {3}".format(i, j, len(trainLoader), loss.float()))
                    losses.append(loss.float())
                    plt.clf()
                    plt.plot(losses)
                    plt.pause(0.01)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            torch.save(self.net, "models/net.pth")
            self.test(testLoader)

    def test(self, loader):
        total = 0.
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            out = self.net(x)
            predict = torch.argmax(out, dim=1)
            total += (predict == y).sum()
        print("Accuracy: {}%".format(total.item() / len(self.testSet) * 100.))


if __name__ == "__main__":
    train = Train()
    train.train()
    print(len(train.trainSet + train.testSet), train.mean, train.std)
