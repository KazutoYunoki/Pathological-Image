from pathological_dataset import PathologicalImage
from make_data_path import make_data_path_list
from image_transform import ImageTransform
from networks import FCNs
from model import train_model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import hydra

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(config_name="parameters")
def main(cfg):
    # 病理画像のデータのリストを取得
    data_dir = make_data_path_list()

    #　データセットの作成
    train_dataset = PathologicalImage(
        file_list=data_dir, transform=ImageTransform(cfg.size), num_pixels=cfg.num_pixels)

    #　動作確認
    print("入力画像サイズ：" + str(train_dataset.__getitem__(0)[0].size()))
    print("教師データサイズ：" + str(train_dataset.__getitem__(0)[1].shape))

    # DataLoaderを作成
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )

    #　動作確認
    batch_iterator = iter(train_dataloader)
    inputs, labels = next(batch_iterator)
    log.info("-----Image and label shape of dataloader-----")
    log.info("入力データ：" + str(inputs.size()))
    log.info("入力ラベル：" + str(labels.shape))

    # GPU初期設定
    # ネットワークモデル(自作FCNs)をimport
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ネットワークをGPUへ
    net = FCNs()
    net.to(device)
    torch.backends.cudnn.benchmark = True
    log.info("-----Constitution of network-----")
    log.info(net)

    # 損失関数の設定
    criterion = nn.MSELoss()

    # 最適化手法の設定
    optimizer = optim.SGD(net.parameters(), lr=cfg.SGD.lr, momentum=cfg.SGD.lr)
    log.info("-----Details of optimizer function-----")
    log.info(optimizer)

    # 損失値を保持するリスト
    train_loss = []

    # 学習
    for epoch in range(cfg.num_epochs):
        log.info("Epoch {} / {} ".format(epoch + 1, cfg.num_epochs))
        log.info("----------")

        #　学習
        train_history = train_model(
            net, train_dataloader, criterion, optimizer)

        # 学習したlossと認識率のリストを作成
        train_loss.append(train_history)

    # figインスタンスとaxインスタンスを作成
    fig_loss, ax_loss = plt.subplots(figsize=(10, 10))
    ax_loss.plot(range(1, cfg.num_epochs + 1, 1),
                 train_loss, label="train_loss")
    ax_loss.set_xlabel("epoch")
    fig_loss.savefig("loss.png")

    # パラメータの保存
    save_path = './pathological.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == "__main__":
    main()
