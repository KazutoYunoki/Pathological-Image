from pathological_dataset import PathologicalImage
from make_data_path import make_data_path_list
from image_transform import ImageTransform
from networks import FCNs
from model import train_model, val_model, test_model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import logging
import hydra

# A logger for this file
log = logging.getLogger(__name__)

# TODO ネットワークにSigmoid関数を通してからlossの下がりが悪すぎる原因の究明


@hydra.main(config_name="parameters")
def main(cfg):
    # 病理画像のデータのリストを取得
    data_dir = make_data_path_list()

    #　データセットの作成
    dataset = PathologicalImage(
        file_list=data_dir, transform=ImageTransform(cfg.size), num_pixels=cfg.num_pixels)

    # 学習用データの枚数を取得
    train_size = int(len(dataset) * cfg.rate)

    # 検証用のデータの枚数を取得
    val_size = len(dataset) - train_size

    # データセットの分割
    train_dataset, val_dataset = data.random_split(
        dataset, [train_size, val_size])

    #　動作確認
    print("入力画像サイズ：" + str(train_dataset.__getitem__(0)[0].size()))
    print("教師データサイズ：" + str(train_dataset.__getitem__(0)[1].shape))

    # 学習用のDataLoaderを作成
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )

    # 検証用のDataLoaderを作成
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False
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
    optimizer = optim.SGD(net.parameters(), lr=cfg.SGD.lr,
                          momentum=cfg.SGD.momentum)
    log.info("-----Details of optimizer function-----")
    log.info(optimizer)

    # 損失値を保持するリスト
    train_loss = []
    val_loss = []

    # 学習
    for epoch in range(cfg.num_epochs):
        log.info("Epoch {} / {} ".format(epoch + 1, cfg.num_epochs))
        log.info("----------")

        #　学習
        train_history = train_model(
            net, train_dataloader, criterion, optimizer)

        # 学習したlossのリストを作成
        train_loss.append(train_history)

        # 検証
        val_history = val_model(net, val_dataloader, criterion)

        #　検証したlossのリスト作成
        val_loss.append(val_history)

    # テストと出力値保存
    test_history = test_model(net, val_dataloader, criterion)

    # figインスタンスとaxインスタンスを作成
    fig_loss, ax_loss = plt.subplots(figsize=(10, 10))
    ax_loss.plot(range(1, cfg.num_epochs + 1, 1),
                 train_loss, label="train_loss")
    ax_loss.plot(range(1, cfg.num_epochs + 1, 1), val_loss, label="val_loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()
    fig_loss.savefig("loss.png")

    # パラメータの保存
    save_path = './pathological.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == "__main__":
    main()
