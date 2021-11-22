"""
Already made
- 기존에 만들어 놓은 COCO Format으로 바꾸는 코드 추가 필요
- mask, data 겹쳐서 확인

New
- mask 겹치는 코드 필요
- 분포 확인
- image crop(resize) to 512x512 size

mask 기준
- dent 를 scratch 위에
- scratch 를 dent 위에
- dent & scratch 라는 label 새로 만들기

"""
##
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

data_dir = "../data/accida_segmentation_dataset_v1"
##
def image_resize():
    pass


def image_crop():
    pass


def combine_mask(msk_dir1, msk_dir2):
    """
    image histogram : http://www.gisdeveloper.co.kr/?p=6634

    dent 의 damage 가 더 critical 한 것이기 때문에 dent mask가 scatch mask를 덮어씌게 할 예정

    overlaped part can be get by & operator
    https://python.plainenglish.io/how-to-find-an-intersection-between-two-matrices-easily-using-numpy-30263373b546
    """
    # mask directory : msk1_dir, msk2_dir
    msk_list1 = os.listdir(msk_dir1)
    msk_list2 = os.listdir(msk_dir2)

    msk_list1.sort()
    msk_list2.sort()

    if(len(msk_list1)!=len(msk_list2)):
        print('The number of masks are different something wrong')
        return 0

    ii = 0
    # for idx, (msk1_name, msk2_name) in enumerate(zip(msk_list1, msk_list2)):
    #     msk1 = cv2.imread(os.path.join(msk_dir1, msk1_name), 0)  # read image as gray scale
    #     msk2 = cv2.imread(os.path.join(msk_dir2, msk2_name), 0)
        # msk1 = np.where(msk1 < 10, 0, msk1)  # 10이하의 값들은 0으로 바꿔줌
        # msk1 = np.where(msk1 > 245, 255, msk1)  # 245이상의 값들은 255으로 바꿔줌
        # msk2 = np.where(msk2 < 10, 0, msk2)  # 10이하의 값들은 0으로 바꿔줌
        # msk2 = np.where(msk2 > 245, 255, msk2)  # 245이상의 값들은 255으로 바꿔줌

    if 1:
        # sample example
        msk1_name = '20190219_10475_20150443_21cf2f5372506a4fb69a550617a66c3d.jpg'
        sample_name = msk1_name
        msk1 = cv2.imread(os.path.join(msk_dir1, sample_name), 0)  # read image as gray scale
        msk2 = cv2.imread(os.path.join(msk_dir2, sample_name), 0)

        msk1 = np.array(msk1)
        msk2 = np.array(msk2)
        msk1 = np.where(msk1 < 10, 0, msk1)
        msk1 = np.where(msk1 > 245, 255, msk1)
        msk2 = np.where(msk2 < 10, 0, msk2)
        msk2 = np.where(msk2 > 245, 255, msk2)


        print(f'mask shape : {msk1.shape}')

    # dent, scratch 어떤걸 위에 올릴지 정하는 부분
        msk_new = np.zeros((msk2.shape[0], msk2.shape[1], 3))  # new mask shape : mask size with 3 channel
        # overlapped area : msk3
        msk3 = msk1 * msk2
        msk3_p = np.where(msk3 != 0) # overlap area to 0
        msk1[msk3_p] = 0 # overlap area process 1
        # msk2[msk3_p] = 0 # overlap area process 2
        msk_new[:, :, 0] = msk2 # msk2 color : (255, 0, 0)
        msk_new[:, :, 2] = msk1 # msk1 color : (0, 0, 255)

        print(f'overlaped1 : {len(np.where(msk_new != (255.0, 0, 255.0))[0])}')  # 이미지 좌표에 접근하는게 지금 헷갈리는 듯?
        print(f'overlaped2 : {len(np.where(msk_new == (255.0, 0, 255.0))[0])}')
        print(f'{msk_new[:,:,0].min()}  {msk_new[:,:,0].max()}')

        # print(f'overlaped1 : {len(np.where(msk1 != 255)[0])}')
        # print(f'overlaped2 : {len(np.where(msk1 == 255)[0])}')

        print(f'new mask shape : {msk_new.shape}')

        # pixel count
        n1 = len(np.where(msk1 != 0)[0])
        n2 = len(np.where(msk2 != 0)[0])



        if ii<10:
            if(n1==0 and n2==0):
                pass
            else:# (n1 != n2):
                print(f'{msk1_name}')
                print(f'n1 {n1}  n2 {n2}')
                ii += 1

                plt.figure(figsize=(10,10))
                plt.subplot(231)
                plt.imshow((msk1 * 255).astype(np.uint8))
                # plt.imshow(msk1)

                plt.subplot(232)
                img_ = cv2.imread(os.path.join(data_dir,'dent','train','images',sample_name))
                img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                plt.imshow(img_)

                plt.subplot(233)
                plt.imshow((msk2 * 255).astype(np.uint8))

                plt.subplot(212)
                # plt.imshow(msk3)
                plt.imshow(msk_new)

                plt.show()






## 실행하는 부분

# for c in ['dent', 'scratch', 'spacing']:
#     for t in ['test', 'valid', 'train']:
#         file_list = os.listdir(os.path.join(data_dir,c,t,'masks'))
#         print(len(file_list))

msk_dir1 = os.path.join(data_dir,'dent','train','masks')
msk_dir2 = os.path.join(data_dir,'scratch','train','masks')

combine_mask(msk_dir1, msk_dir2)








