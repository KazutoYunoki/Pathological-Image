import pathlib
import os.path as osp
import pandas as pd
import csv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import torch
# 教師データ（ダーモスコピー上の切片色）を扱う関数群


def calculate_rgb_ave(img):
    """
    画像から長軸RGBの平均値を計算してnp.arrayで返す関数
    Parameters
    ----------
    img : RGB画像
    Returns
    ----------
    rgb_ave : np.array型
        RGBの1次元プロファイル（例1000×3)
    """

    # 切り取った個所をRGB別に取得
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]

    # RGB別に列方向対する平均値を取得
    ave_r = np.average(img_r, axis=0)
    ave_g = np.average(img_g, axis=0)
    ave_b = np.average(img_b, axis=0)

    # それぞれの色情報を正規化する(0-1)
    for i in range(ave_r.size):

        ave_r[i] = float(ave_r[i]) / float(255)
        ave_g[i] = float(ave_g[i]) / float(255)
        ave_b[i] = float(ave_b[i]) / float(255)

    # R,G,Bの色情報の平均を1つにまとめる(例 1000*3チャンネル)
    rgb_ave = []
    for i in range(ave_r.size):
        rgb_ave.append([ave_r[i], ave_g[i], ave_b[i]])

    # np.array型（float）に変換して保存
    rgb_ave = np.array(rgb_ave, dtype=np.float32)

    return rgb_ave

# 指定したピクセル数に揃えるための関数（Paddingとか）

# TODO　ラベルのピクセル数のとり方の処理（現在：0パディング）


def align_pixels(color_arr, pixel_num):
    """
    指定したピクセル数でカラープロファイルを整形する関数
    Parameters
    ----------
    color_arr : np.array型
        2次元のカラープロファイル（例1500×3）
    pixel_num : int
        整形したいピクセル数を指定
    Returns
    ----------
    color_arr : np.array型
        整形した後のカラープロファイル
    """

    # 色の配列の長さを取得
    color_len = len(color_arr)

    # 余ったPixelを計算
    extra_pixel = color_len - pixel_num

    # 余ったPixelがある場合、真ん中のpixel_numだけ取得
    if(extra_pixel >= 0):
        h = int(extra_pixel / 2)
        color_arr = color_arr[h:h+pixel_num]
    else:
        for i in range(color_len, pixel_num, 2):
            color_arr = np.insert(color_arr, 0, np.array([0, 0, 0]), axis=0)
            if(len(color_arr) == pixel_num):
                break
            color_arr = np.insert(
                color_arr, color_arr.shape[0], np.array([0, 0, 0]), axis=0)

    #　深層学習内では[チャンネル数]、[高さ]、[幅]なので軸を入れ替える
    color_arr = color_arr.transpose(1, 0)

    return color_arr


def make_crop_img_list():
    """
    青いボックスで切り取ったcropImageのPathリストを取得する関数
    Return
    ---------
    path_list : list
        crop imgのファイルを格納したリスト
    """
    # 現在のディレクトリを取得(Pathological-Image)
    current_dir = current_dir = pathlib.Path(__file__).resolve().parent
    current_dir = str(current_dir)

    # データセットのルートパス
    rootpath = current_dir + '/data/'
    target_path = osp.join(rootpath + "/**/outputs/crop_img.png")

    path_list = []

    # globを利用してファイルパス取得
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


if __name__ == "__main__":
    im = Image.open('./data/9497/outputs/crop_img.png')
    #im = np.array(im)
    im = np.array(im)

    result = calculate_rgb_ave(im)
    print(result)
