import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import transforms
from PIL import Image, ImageOps
from torchsummary import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)


# use_cuda = torch.cuda.is_available()
# print("use_cuda:", use_cuda)
#
# device = torch.device("cuda" if use_cuda else "cpu")
#
# model = 0
# model = Net().to(device)
#
# print(device)
# print(model)
# print(summary(model, (1, 28, 28)))
#
# #学習モデルのロード
# model.load_state_dict(torch.load('mnist_cnn.pt', map_location=lambda storage, loc: storage))
# model = model.eval()
#
# #画像ファイルを読み込む
# path = 'mnist_2.jpg'
# image = Image.open(path)
# image = ImageOps.invert(image)
# image = image.convert('L').resize((28,28))
#
# #データの前処理の定義
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,),(0.3081))
# ])
#
# #元のモデルに合わせて次元を追加
# image = transform(image).unsqueeze(0)
#
# #予測
# output = model(image.to(device))
# _, prediction = torch.max(output, 1)
#
# #結果を出力
# print('{}->result = '.format(path) + str(prediction[0].item()))