from pathological_dataset import PathologicalImage
from make_data_path import make_data_path_list
from image_transform import ImageTransform
import torch.utils.data as data
data_dir = make_data_path_list()

size = 224
num_pixels = 1024

#　データセットの作成
dataset = PathologicalImage(
    file_list=data_dir, transform=ImageTransform(size), num_pixels=num_pixels)

print(len(dataset))

num_datasets = len(dataset)

train_size = int(len(dataset) * 0.85)

val_size = num_datasets - train_size

# データセットの分割
train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])


print(len(train_dataset))
print(len(val_dataset))
