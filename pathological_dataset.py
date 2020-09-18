import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
from image_transform import ImageTransform
from make_teacher_data import make_crop_img_list, calculate_rgb_ave, align_pixels


class PathologicalImage(data.Dataset):
    """
    病理画像のDatasetクラス。PytorchのDatasetクラスを継承

    Attributes
    ----------
    file_list : リスト
        画像のパスを格納したリスト

    transform : object
        画像の前処理クラスのインスタンス

    phase: 'train' or 'val'
        学習用か訓練用かを設定
    """

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  # ファイルパスのリスト
        self.transform = transform  # 画像の前処理のインスタンス
        self.phase = phase

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.file_list)

    def __getitem__(self, index):
        """
        前処理をした画像のTensor形式のデータとラベルを取得
        """

        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅][RGB]

        # 画像の前処理を実施
        img_transformed = self.transform(img, self.phase)

        # 教師データのファイルパスを取得
        teacher_list = make_crop_img_list()

        # RGBの平均値プロファイルを取得
        rgb_ave = calculate_rgb_ave(teacher_list[index])

        # 指定したピクセル数で整形
        color_arr = align_pixels(rgb_ave, 1000)

        # labelにダーモスコピー上の色情報（1000×3）
        label = color_arr

        return img_transformed, label


def main():
    # 画像の読み込みと表示
    img_path = './data/9497/resize-070327.jpg'
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()

    size = (512, 1024)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = ImageTransform(size, mean, std)
    img_transform = transform(img, phase='train')
    print(img_transform.shape)

    img_transformed = img_transform.numpy().transpose((1, 2, 0))
    img_transformed = np.clip(img_transformed, 0, 1)
    plt.imshow(img_transformed)
    plt.show()


if __name__ == "__main__":
    main()