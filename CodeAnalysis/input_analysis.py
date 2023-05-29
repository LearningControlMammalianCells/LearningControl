import tifffile
import cv2
import numpy as np
import pandas as pd
from functions import *
import matplotlib.pyplot as plt

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
            img = 't' + zt + t +'xy' + zpos + chamber + 'c3.tif'
            #print(img)
            ch1.append(f'{folder}/{img}')
        except:
            kkk=0

    return ch1

if __name__ == '__main__':

    folder = "../Experiments"
    exp = 'nd08'
    folder = f'{folder}/{exp}'
    chamber = 3
    format = 't000001xy1c1.tif'
    data = pd.read_csv(f'{folder}/Dataset_{exp}.csv')
    data = data.drop(columns='Unnamed: 0')
    u = data[data.columns[-1]]

    file_name = extract_file_name(folder, format, chamber, t=1)
    img_raw = tifffile.imread(file_name)
    img_raw0 = (img_raw - np.mean(img_raw)) / np.std(img_raw)
    roi = cv2.selectROI(img_raw0)

    ch1 = extract_paths(folder, format, chamber, Tf=193)

    imgs = [imread(f) for f in ch1]
    imgs = [img_raw[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] for img_raw in imgs]

    red = []
    for img in imgs:
        red.append(np.sum(img))
    red = np.array(red)
    min_r = np.min(red)
    max_r = np.max(red)
    red = (red-min_r)/(max_r - min_r)


    fig, ax = plt.subplots(figsize=(10, 2))
    plt.step(np.arange(len(u))*15, u,linewidth = 2,where = 'post',label = 'Theoretical')
    plt.plot(np.arange(len(u))*15, red,linewidth =2, c =  '#F16996',label ='Real')

    plt.grid(linestyle='dashed', alpha=0.7)
    plt.ylabel('Inducer [AU]')
    plt.xlabel('Time [min]')
    plt.legend(loc = 'upper left')
    plt.savefig(f'{folder}/immagini/input.png', bbox_inches='tight')
    plt.show()

