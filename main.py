"""
[Socar X AIFFEL] Car Damage Detection

making sample prototype

"""

"""
Record
2021.10.28
- Image check code needed


2021.10.19
- 각 모듈 별로 구현을 하여, 입력 영상을 주면 원하는 형태의 최종 결과 나오는 것 확인
  - 모듈별로 다른 알고리즘을 쓸 예정..
- MLOPS나 DevOPS로 하는게 그런 모델들 올려서 돌리게 해주는거아닌가? 
  -> 그럼 이 큰 프로젝트 단위로 만드는게 썩 중요한게 아닐지도?
"""
# from src import 만든 모듈명
from src import *
from src import car_localize, carcomponent_classify
from src import globaldamage_classify,localdamage_classify

import matplotlib.pyplot as plt
import numpy as np
import cv2

if __name__ == '__main__':

    print('in main')
    img_dir = ['./data/example.jpeg']

    imageA = cv2.imread(img_dir[0])
    imageA = cv2.cvtColor(imageA,cv2.COLOR_BGR2RGB)
    # plt.imshow(imageA)
    # plt.waitforbuttonpress(0)
    # plt.close()

    car_localize.detect_car(imageA)
