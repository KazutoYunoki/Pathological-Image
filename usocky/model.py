from tqdm import tqdm
from pathlib import Path
import torch
import csv
import logging

# A logger for this file
log = logging.getLogger(__name__)


def train_model(net, train_dataloader, criterion, optimizer):
    """
    学習させるための関数
    Parameters
    ----------
    net :
        ネットワークモデル
    train_dataloader :
        学習用のデータローダ
    criterion :
        損失関数
    optimizer :
        最適化手法
    Returns
    -------
    epoch_loss : double
        エポックあたりのloss値
    """

    # GPU初期設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.train()

    epoch_loss = 0.0

    for inputs, labels in tqdm(train_dataloader, leave=False):

        # GPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            # ネットワークにデータ入力
            outputs = net(inputs)

            # 損失値の計算
            loss = criterion(outputs, labels)

            # バックプロパゲーションで重み更新
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)

    # epochごとのlossを計算
    epoch_loss = epoch_loss / len(train_dataloader.dataset)

    log.info("Train Loss: {:.4f}".format(epoch_loss))

    return epoch_loss


def val_model(net, val_dataloader, criterion):
    """
    モデルで検証させる関数
    Parameters
    ----------
    net :
        ネットワークモデル
    val_dataloader :
        検証用のデータローダ
    criterion :
        損失関数
    Returns
    -------
    epoch_loss :
        エポックあたりの損失値
    """
    # GPU初期設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 評価モードへ変更
    net.eval()

    epoch_loss = 0.0

    for inputs, labels in tqdm(val_dataloader, leave=False):

        # GPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

            loss = criterion(outputs, labels)

        epoch_loss += loss.item() * inputs.size(0)

    # epochごとのlossの計算
    epoch_loss = epoch_loss / len(val_dataloader.dataset)

    log.info("Test Loss: {:.4f}".format(epoch_loss))

    return epoch_loss


def test_model(net, test_dataloader, criterion):
    """
    モデルをテストさせる関数
    Parameters
    ----------
    net :
        ネットワークモデル
    test_dataloader :
        テスト用のデータローダ
    criterion :
        損失関数
    Returns
    -------
    epoch_loss :
        エポックあたりの損失値
    """
    # GPU初期設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 評価モードへ変更
    net.eval()

    epoch_loss = 0.0

    for inputs, labels in tqdm(test_dataloader, leave=False):

        # GPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

            loss = criterion(outputs, labels)

        epoch_loss += loss.item() * inputs.size(0)
        # outputとlabelをcsvファイルに出力
        save_output_date(outputs, labels, phase='val')

    # epochごとのlossの計算
    epoch_loss = epoch_loss / len(test_dataloader.dataset)

    log.info("Test Loss: {:.4f}".format(epoch_loss))

    return epoch_loss


def save_output_date(output_data, label_data, phase):
    """
    ネットワークの出力結果とラベルをCSVファイルに保存する関数
    Parameters
    ----------
    output_data: tensor
        ネットワークの出力結果
    label_data: tensor
        ラベルデータ
    phase : train or val
        学習データか検証データか指定
    """
    # outputsとlabelsをTensorからnumpyへ変更
    outputs = output_data.cpu().numpy()
    labels = label_data.cpu().numpy()

    # 出力先のディレクトリを作成
    output_dir = Path('./' + phase + '_output_color')
    output_dir.mkdir(exist_ok=True)

    # CSVファイルを用意して予測結果とラベルを書き込み
    with open(str(output_dir) + '/color.csv', 'a') as f:
        writer = csv.writer(f)
        for i in range(len(outputs)):
            writer.writerow(['予測結果:' + str(i)])
            writer.writerows(outputs[i])
            writer.writerow(['ラベル:' + str(i)])
            writer.writerows(labels[i])
