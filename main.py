from pathological_dataset import PathologicalImage
from make_data_path import make_data_path_list
from image_transform import ImageTransform


def main():
    # 病理画像のデータのリストを取得
    data_dir = make_data_path_list()

    size = (512, 1024)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_dataset = PathologicalImage(
        file_list=data_dir, transform=ImageTransform(size, mean, std))

    # 動作確認
    for i in range(len(data_dir)):

        print(train_dataset.__getitem__(i)[0].size())
        print(train_dataset.__getitem__(i)[1].shape)


if __name__ == "__main__":
    main()
