from torchvision import transforms


class ImageTransform():
    """
    画像の前処理を行うクラス
    """

    def __init__(self, resize):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                # transforms.RandomHorizontalFlip()
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


# TODO ラベルの前処理をどうするか？　データオーグメンテーションなど
class LabelTransform():
    """
    ラベルの前処理を行うクラス
    """

    def __init__(self):
        self.label_transform = {
            'train': transforms.Compose([
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.ToTensor()
            ])
        }

    def __call__(self, label, phase='train'):
        return self.label_transform[phase](label)
