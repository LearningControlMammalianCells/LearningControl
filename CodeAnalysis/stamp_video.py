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

def save_video(folder, ch1,ch,chamber, withmask):
    files = []
    flag = 0
    ims = []
    imgs = []
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()
    cnt = 0
    masks = np.load(f'{folder}/masks/{chamber}.npy')

    for name in ch1:
        if ch ==2:
            name = name.replace('c1', 'c2')
        files.append(name)
        data = tifffile.imread(name)


        imgs.append(data)
        if ch==1:
            cmap = 'gray'
        elif ch==2:
            cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.35, reverse=True, as_cmap=True)

        if ch==1 and withmask ==1:
            roi = np.load(f'{folder}/roi/roi_{chamber}.npy')
            data = data[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            maski = masks[cnt]
            imgcell = (data * maski) / 10
            data = data + imgcell
        im = ax.imshow(data, animated=True, cmap=cmap)
        time = name.split('x')[0]
        time = cnt*15
        title = ax.text(0.5, 1.05, f'CH: {chamber} Min : {time}',
                        size=50,
                        ha="center", transform=ax.transAxes, )
        ims.append([im, title])
        cnt = cnt+1
    # Writer = animation.writers['pillow']
    # ani = animation.ArtistAnimation(fig, ims,interval = 500)
    # plt.show()
    matplotlib.rcParams[
        'animation.ffmpeg_path'] = 'C:\\Users\\a.dehoffer\\Anaconda3\\pkgs\\ffmpeg-4.3.1-ha925a31_0\\Library\\bin\\ffmpeg.exe'
    writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)

    ani = animation.ArtistAnimation(fig, ims, interval=500)
    if ch==1 and withmask==1:
        ani.save(f'{folder}/immagini/{chamber}_ch{ch}_mask.avi', writer=writer)
    else:
        ani.save(f'{folder}/immagini/{chamber}_ch{ch}.avi', writer=writer)
    fig.canvas.draw()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ehi')

    folder = '../Experiments/nd08'
    format = 't000001xy1c1.tif'
    Tf = 193
    if not os.path.isdir(f'{folder}/immagini'):
        os.mkdir(f'{folder}/immagini')
    for chamber in range(2,6):
        for channel in [1,2]:
            print(chamber,channel)
            ch1 = extract_paths(folder, format, chamber, Tf)
            save_video(folder = folder, ch1 = ch1, chamber = chamber, ch=channel,withmask =1)
    print('done')


