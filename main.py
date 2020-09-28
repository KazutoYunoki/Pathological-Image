from pathological_dataset import PathologicalImage
from make_data_path import make_data_path_list
from image_transform import ImageTransform
from networks import FCNs
from model import train_model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    # 病理画像のデータのリストを取得
    data_dir = make_data_path_list()

    size = (224, 224)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    #　データセットの作成
    train_dataset = PathologicalImage(
        file_list=data_dir, transform=ImageTransform(size), num_pixels=1024)

    #　動作確認
    print("入力画像サイズ：" + str(train_dataset.__getitem__(0)[0].size()))
    print("教師データサイズ：" + str(train_dataset.__getitem__(0)[1].shape))

    # ミニバッチのサイズを指定
    batch_size = 8

    # DataLoaderを作成
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    #　動作確認
    batch_iterator = iter(train_dataloader)
    inputs, labels = next(batch_iterator)
    print("-----DataLoaderの画像とラベルの形状-----")
    print("入力データ：" + str(inputs.size()))
    print("入力ラベル：" + str(labels.shape))

    # GPU初期設定
    # ネットワークモデル(自作FCNs)をimport
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ネットワークをGPUへ
    net = FCNs()
    net.to(device)
    torch.backends.cudnn.benchmark = True
    print("-----ネットワークの構成-----")
    print(net)

    # 損失関数の設定
    criterion = nn.MSELoss()

    # 最適化手法の設定
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 損失値を保持するリスト
    train_loss = []

    # 学習
    num_epochs = 2
    for epoch in range(num_epochs):
        print("Epoch {} / {} ".format(epoch + 1, num_epochs))
        print("----------")

        #　学習
        train_history = train_model(
            net, train_dataloader, criterion, optimizer)

        # 学習したlossと認識率のリストを作成
        train_loss.append(train_history)

    # figインスタンスとaxインスタンスを作成
    fig_loss, ax_loss = plt.subplots(figsize=(10, 10))
    ax_loss.plot(range(1, num_epochs + 1, 1),
                 train_loss, label="train_loss")
    ax_loss.set_xlabel("epoch")
    fig_loss.savefig(loss.png)

    # パラメータの保存
    save_path = './pathological.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == "__main__":
    main()
