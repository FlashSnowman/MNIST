import torch
from TrainNet import Train
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


class TNet:
    net = torch.load("models/net.pth")
    train = Train()

    def testPic(self):
        image = Image.open(r"F:\image\8.jpg")
        # image = transforms.Grayscale()(image)
        image = image.convert("L")
        # image = transforms.Resize(28)(image)
        image = image.resize((28, 28))
        image = transforms.ToTensor()(image)
        image = 1. - image
        image = transforms.Normalize((TNet.train.mean,), (TNet.train.std,))(image)
        image = image.unsqueeze(0)
        out = TNet.net(image.cuda())
        predict = out.argmax(1)
        print("AI的预测值为：{}".format(predict.item()))

    def testNet(self, loader):
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            out = TNet.net(x)
            datas = (x * TNet.train.std + TNet.train.mean) * 255.
            datas = datas.cpu().numpy().astype(np.uint8)
            for i in range(x.shape[0]):
                data = datas[i].squeeze()
                predict = out.argmax(1)[i]
                plt.clf()
                plt.subplot(1, 2, 1)
                plt.imshow(data)
                plt.subplot(1, 2, 2)
                plt.text(0.35, 0.35, predict.item(), fontsize=100)
                plt.pause(1)


if __name__ == "__main__":
    test = TNet()
    test.testPic()
    test.testNet(test.train.data_loader()[1])
