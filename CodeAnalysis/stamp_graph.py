import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import sys
import os

if __name__ == '__main__':
    folder = "C:/Users/a.dehoffer/PycharmProjects/Experiments"
    exp = 'nd08'
    folder = f'{folder}/{exp}'

    data = pd.read_csv(f'{folder}/Dataset_{exp}.csv')
    data = data.drop(columns='Unnamed: 0')

    data = data.fillna(method="ffill")
    chambers = data.columns
    chambers = list(chambers)
    fig, ax = plt.subplots(figsize=(10, 3))
    plt.axhline(y=0.5, c='black', alpha=0.7, linestyle='dashed')

    fluos = []
    u = data[chambers[-1]].values
    print(u)
    for chamber in chambers[:-1]:
        fluo = data[chamber].values

        norm = np.mean(fluo[:12])
        fluo = fluo / norm
        #fluo = savgol_filter(fluo, 11, 3)
        fluos.append(fluo)
        plt.plot(np.arange(0,len(fluo))*15,fluo, label=int(chamber)+1,linewidth = 2)

    #plt.plot(np.mean(fluos, axis=0), c='black', linewidth=3, alpha=0.7)

    plt.grid(linestyle='dashed', alpha=0.7)
    plt.legend(title= 'Chamber:')
    plt.xlabel('Time [min]')
    plt.ylabel('Fluorescence [AU]')
    plt.xlim((0,len(fluo)*15))
    plt.savefig(f'{folder}/immagini/graph.png',bbox_inches='tight')

    #plt.show()

    fig, ax = plt.subplots(figsize=(10, 1))
    plt.step(np.arange(0,len(u))*15, u,linewidth = 3, c =  '#F16996')
    plt.grid(linestyle='dashed', alpha=0.7)
    plt.yticks(ticks=[0, 1], labels=['MEDIUM', 'INDUCER'], fontsize=10, horizontalalignment='center', rotation=90,
               rotation_mode='anchor')
    plt.xlabel('Time [min]')
    plt.xlim((0, len(fluo) * 15))
    plt.savefig(f'{folder}/immagini/u.png',bbox_inches='tight')
    #plt.show()