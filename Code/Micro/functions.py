import numpy as np
import cv2
import tifffile
from cellpose import models
from cellpose.io import imread
import matplotlib.pyplot as plt

def extract_file_name(folder,format, chamber,t):

    time, pos = format.split('xy')
    pos = pos.split('c1')[0]
    zeros_pos = pos.count('0') + 1
    zeros_time = time.count('0') + 1
    chamber = str(chamber)
    nc_c = len(chamber)
    zpos = '0' * (zeros_pos - nc_c)
    t = str(t)
    zt = '0' * (zeros_time - len(t))
    img = 't' + zt + t + 'xy' + zpos + chamber + 'c1.tif'
    img = f'{folder}/{img}'
    return img


def cropping(file_name):

    img_raw = tifffile.imread(file_name)
    img_raw0 = (img_raw - np.mean(img_raw)) / np.std(img_raw)
    roi = cv2.selectROI(img_raw0)


    return roi


def segment(file_name, roi, diameter = 40):

    img = file_name


    model = models.Cellpose(gpu=False, model_type='cyto')

    img = imread(img)
    img = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

    mask, flow, style, diams = model.eval(img, diameter=diameter, channels=[0, 0],
               flow_threshold=0.4, do_3D=False)


    return mask

def fluo_eval(mask,file_name,roi):


    data2 = tifffile.imread(file_name)
    data = data2.astype(int)
    data = data[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

    maski = mask


    maski[maski > 0] = 1
    maski_bg = np.logical_not(maski).astype(int)

    img_cell = data * maski
    img_bg = data * maski_bg

    cell = np.sum(img_cell) / np.sum(maski)
    bg = np.sum(img_bg) / np.sum(maski_bg)

    fluo = cell - bg
    return fluo

def lv_fuoco(file_name,roi):

    data1 = tifffile.imread(file_name)
    data1 = data1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    laplacian_var = cv2.Laplacian(data1, cv2.CV_64F).var()
    return laplacian_var

def select_diameter(file_name,roi):

    img = file_name
    model = models.Cellpose(gpu=False, model_type='cyto')

    img = imread(img)
    img = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    flag = 0

    while flag ==0:
        print('Choose diameter (hint :50)')
        diameter = int(input())
        mask, flow, style, diams = model.eval(img, diameter=diameter, channels=[0, 0],
                                              flow_threshold=0.4, do_3D=False)

        fig = plt.figure(figsize=(5, 5))

        imgcell = (img * mask) / 10
        imgcell = img + imgcell
        plt.imshow(imgcell, cmap='gray')
        plt.show()

        print('Ok? 0 No, 1 Yes')
        flag = int(input())
    return diameter



