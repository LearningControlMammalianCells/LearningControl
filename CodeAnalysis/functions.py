import numpy as np
import cv2
import tifffile
from cellpose import models
from cellpose.io import imread
import matplotlib.pyplot as plt
import scipy

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


def cropping(folder,chamber, file_name):

    img_raw = tifffile.imread(file_name)
    img_raw0 = (img_raw - np.mean(img_raw)) / np.std(img_raw)
    roi = cv2.selectROI(img_raw0)


    np.save(f'{folder}/roi/roi_{chamber}.npy', roi)
    return roi


def segment(folder, chamber,  ch1, roi, diameter = 40):

    model = models.Cellpose(gpu=False, model_type='cyto')

    imgs = [imread(f) for f in ch1]


    imgs = [img_raw[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] for img_raw in imgs]


    masks, flow, style, diams = model.eval(imgs, diameter=diameter, channels=[0, 0],
               flow_threshold=0.4, do_3D=False)

    np.save(f'{folder}/masks/{chamber}.npy',masks)
    return masks



def fluo_eval(ch1,masks,roi):
    Ts = 15

    fuoco = np.ones(len(ch1))
    fluos = []
    for idx, name in enumerate(ch1):

        idx = np.copy(idx)
        data1 = tifffile.imread(name)
        data1 = data1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        laplacian_var = cv2.Laplacian(data1, cv2.CV_64F).var()
        if idx == 0:
            initial_fuoco = laplacian_var
            th = initial_fuoco / 100 * 80

        if (laplacian_var) < th:
            fuoco[idx] = 0

        name2 = name.replace('c1', 'c2')
        data = tifffile.imread(name2)
        data = data.astype(int)
        data = data[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        maski = masks[idx]

        maski[maski > 0] = 1
        maski_bg = np.logical_not(maski).astype(int)
        img_cell = data * maski
        img_bg = data * maski_bg
        cell = np.sum(img_cell) / np.sum(maski)
        bg = np.sum(img_bg) / np.sum(maski_bg)
        fluo = cell - bg
        fluos.append(fluo)

    for i in range(len(fuoco)):
        if fuoco[i]==0:
            fluos[i]=fluos[i-1]

    return fluos

def lv_fuoco(file_name,roi):

    data1 = tifffile.imread(file_name)
    data1 = data1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    laplacian_var = cv2.Laplacian(data1, cv2.CV_64F).var()
    return laplacian_var

def select_diameter(folder,chamber, file_name,roi):

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
    np.save(f'{folder}/roi/diameter_{chamber}.npy', diameter)
    return diameter

def extract_paths(folder,format,chamber,Tf):

    ch1 = []
    time, pos = format.split('xy')
    pos = pos.split('c1')[0]
    zeros_pos = pos.count('0') +1
    zeros_time = time.count('0') +1
    chamber = str(chamber)
    nc_c = len(chamber)
    zpos = '0' * (zeros_pos - nc_c)


    for t in range(1,Tf+1): #problema qui
        try:
            t = str(t)
            zt = '0' * (zeros_time - len(t))
            img = 't' + zt + t +'xy' + zpos + chamber + 'c1.tif'
            #print(img)
            ch1.append(f'{folder}/{img}')
        except:
            kkk=0

    return ch1

def extract_u(path_mat, id = 'Lorena'):
    data_mat = scipy.io.loadmat(path_mat)
    if id =='Lorena':
        u = data_mat['vton']
        u = u[0]
        u = list(map(lambda x: 0 if x == 0 else 1, u))
    else:
        u = data_mat['in']
        u = u[0]
        u = np.array(list(map(lambda x: 0 if x == 1 else 1, u)))


    #selected_chamber = data_mat['selected_field'][0][0]
    return u
