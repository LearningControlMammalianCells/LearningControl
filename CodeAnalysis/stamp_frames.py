from cellpose import models
from cellpose.io import imread
from cellpose import plot
import matplotlib.pyplot as plt
import os
import numpy as np
import tifffile
import cv2
import glob
import matplotlib.animation as animation
from IPython.display import HTML
import ffmpeg
import matplotlib
import seaborn as sns
from functions import *

def stamp_frames(folder, ch1, chamber, ch, withmask=0):
    iles = []
    flag = 0
    ims = []
    imgs = []

    cnt = 0

    for name in ch1:
        if ch == 2:
            name = name.replace('c1', 'c2')

        data = tifffile.imread(name)

        imgs.append(data)
        if ch == 1:
            cmap = 'gray'
        elif ch == 2:
            cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.35, reverse=True, as_cmap=True)

        '''if ch==1 and withmask ==1:
            roi = np.load(f'nd02_CHOTETOFFUbEGFP_OpenLoop/roi{cameretta}.npy')
            data = data[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            maski = np.load(f'nd02_CHOTETOFFUbEGFP_OpenLoop/masks/{cameretta}/mask{cnt}.npy')
            imgcell = (data * maski) / 10
            data = data + imgcell'''
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_axis_off()
        plt.imshow(data, cmap=cmap)

        plt.savefig(f'{folder}/immagini/{chamber}_ch{ch}_{cnt}.png')
        cnt = cnt+1






# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print('ehi')

    folder = '../Experiments/nd07'
    format = 't000001xy1c1.tif'
    Tf = 10
    if not os.path.isdir(f'{folder}/immagini'):
        os.mkdir(f'{folder}/immagini')
    for chamber in range(1, 6):
        for channel in [1, 2]:
            print(chamber, channel)
            ch1 = extract_paths(folder, format, chamber, Tf)
            stamp_frames(folder=folder, ch1=ch1, chamber=chamber, ch=channel, withmask=0)
        break
    print('done')




