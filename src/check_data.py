import cv2
import numpy as np
import sys
import os
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

def read_data(dir_path = os.getcwd()+'/data/segmentation/'):

    print(os.getcwd())
    print(os.path.dirname(os.path.realpath(__file__)))

 # 확인해보고 싶은 데이터 CLASS
    cls = 'dent'
    # cls = 'scratch'
    # cls = 'spacing'
    img_path = dir_path+cls+'/valid/images/'
    mask_path = dir_path + cls + '/valid/masks/'

 # 해당 directory file 긁어옴
    image_list = os.listdir(img_path)
    mask_list = os.listdir(mask_path)
    print(os.listdir(img_path))
    print(len(os.listdir(img_path)))

    for idx, (image, mask) in enumerate(zip(image_list, mask_list)):
      # 일단 한장
        if idx<1:
            print(f'{idx+1} 번째 : {image}, {mask}')

            img = cv2.imread(img_path+image)  # 오른쪽 사진
            msk = cv2.imread(mask_path+mask)  # 왼쪽 사진

            print(f'image shape : {img.shape}, mask shape : {msk.shape}')
            img_show = cv2.hconcat([img,msk])
            print(img_show.shape)

            # mask 부분
            _msk = msk[:,:,0]
            _msk = cv2.bitwise_not(_msk)
            _msk_ = np.zeros(_msk.shape)

            color_mask = cv2.applyColorMap(_msk, cv2.COLORMAP_JET)
            # mask 색깔 단일로 만들어야 보기 좋음
            color_mask[:, :, 1] = np.zeros(_msk.shape)
            color_mask[:, :, 2] = np.zeros(_msk.shape)
            print(f'color mask : {color_mask.shape}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_show = cv2.addWeighted(img, 1, color_mask, 0.8, 0.0)

            plt.imshow(img_show)
            plt.waitforbuttonpress(-1)
            plt.close()


