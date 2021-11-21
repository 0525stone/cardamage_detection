"""
Already made
- 기존에 만들어 놓은 COCO Format으로 바꾸는 코드 추가 필요
- mask, data 겹쳐서 확인

New
- mask 겹치는 코드 필요
- 분포 확인
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





def combine_mask(msk_dir1, msk_dir2):
    """
    image histogram : http://www.gisdeveloper.co.kr/?p=6634

    dent 의 damage 가 더 critical 한 것이기 때문에 dent mask가 scatch mask를 덮어씌게 할 예정

    overlaped part can be get by & operator
    """
    # mask directory : msk1_dir, msk2_dir
    msk_list1 = os.listdir(msk_dir1)
    msk_list2 = os.listdir(msk_dir2)

    msk_list1.sort()
    msk_list2.sort()

    if(len(msk_list1)!=len(msk_list2)):
        print('The number of masks are different')
        return 0

    ii = 0
    for idx, (msk1_name, msk2_name) in enumerate(zip(msk_list1, msk_list2)):
        msk1 = cv2.imread(os.path.join(msk_dir1, msk1_name), 0) # read image as gray scale
        msk2 = cv2.imread(os.path.join(msk_dir2, msk2_name), 0)

        msk1 = np.array(msk1)
        msk2 = np.array(msk2)

        n1 = len(np.where(msk1 != 0)[0])
        n2 = len(np.where(msk2 != 0)[0])

        # if ii<10:
        if(n1==0 and n2==0):
            pass
        else:# (n1 != n2):
            print(f'n1 {n1}  n2 {n2}')
            ii += 1

            plt.figure()
            plt.subplot(121)
            plt.imshow(msk1)

            plt.subplot(122)
            plt.imshow(msk2)

            plt.show()



    # else:
        #     break


        # # check histogram not completed
        # print(f'{msk1_name}, {msk2_name}')

        #
        # plt.subplots(121)
        # plt.hist(msk1.ravel(), 256, [0,256])
        #
        # plt.subplots(122)
        # plt.hist(msk2.ravel(), 256, [0, 256])
        #
        # plt.show()





## 실행하는 부분

# for c in ['dent', 'scratch', 'spacing']:
#     for t in ['test', 'valid', 'train']:
#         file_list = os.listdir(os.path.join(data_dir,c,t,'masks'))
#         print(len(file_list))

msk_dir1 = os.path.join(data_dir,'dent','train','masks')
msk_dir2 = os.path.join(data_dir,'scratch','train','masks')

combine_mask(msk_dir1, msk_dir2)








