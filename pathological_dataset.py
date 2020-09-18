from PIL import Image
import matplotlib.pyplot as plt
from image_transform import ImageTransform
import numpy as np


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
