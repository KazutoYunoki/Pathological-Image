from pathological_dataset import PathologicalImage
from make_data_path import make_data_path_list
from image_transform import ImageTransform

import torch


def main():
    # 病理画像のデータのリストを取得
    data_dir = make_data_path_list()

    size = (224, 224)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_dataset = PathologicalImage(
        file_list=data_dir, transform=ImageTransform(size, mean, std), num_pixels=1024)

    #　動作確認
    print(train_dataset.__getitem__(0)[0].size())
    print(train_dataset.__getitem__(0)[1].shape)

    # ミニバッチのサイズを指定
    batch_size = 8

    # DataLoaderを作成
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    #　動作確認
    batch_iterator = iter(train_dataloader)
    inputs, labels = next(batch_iterator)
    print(inputs.size())
    print(labels.shape)


if __name__ == "__main__":
    main()
