from flask import Flask

import torch
from torchvision import transforms
from PIL import Image, ImageOps

import json

from predict import Net

app = Flask(__name__)

@app.route('/predict/<path:path>')
def func(path):
    # path = 'r' + path
    print(path)
    use_cuda = torch.cuda.is_available()
    print("use_cuda:", use_cuda)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)

    # 学習モデルのロード
    model.load_state_dict(torch.load('mnist_cnn.pt', map_location=lambda storage, loc: storage))
    model = model.eval()

    # 画像ファイルを読み込む
    image = Image.open(path)
    image = ImageOps.invert(image)
    image = image.convert('L').resize((28, 28))

    # データの前処理の定義
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081))
    ])

    # 元のモデルに合わせて次元を追加
    image = transform(image).unsqueeze(0)

    # 予測
    output = model(image.to(device))
    _, prediction = torch.max(output, 1)

    result = {'prediction': prediction[0].item()}
    result = json.dumps(result, indent=2, ensure_ascii=False)
    return result

if __name__ == '__main__':
    app.run()