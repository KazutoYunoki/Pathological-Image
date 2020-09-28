from tqdm import tqdm
import torch

# TODO 検証用の関数の実装


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

    print("Train Loss: {:.4f}".format(epoch_loss))

    return epoch_loss
