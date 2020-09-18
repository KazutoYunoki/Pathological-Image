import pathlib
import os.path as osp
import glob
# 病理画像のファイルパスのリストを作成する


def make_data_path_list(phase="train"):
    """
    データのパスを格納したリストを作成

    Parameters
    ----------
    phase : 'train' or 'val'
        訓練データか検証データかを指定

    Returns
    ----------
    path_list : list
        画像データのパスを格納したリスト
    """

    # 現在のディレクトリを取得(Pathological-Image)
    current_dir = current_dir = pathlib.Path(__file__).resolve().parent
    current_dir = str(current_dir)

    # データセットのルートパス
    rootpath = current_dir + '/data/'
    target_path = osp.join(rootpath + "/**/resize*")
    print(target_path)

    path_list = []

    # globを利用してファイルパス取得
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


if __name__ == "__main__":
    list = make_data_path_list(phase='train')
    print(len(list))
