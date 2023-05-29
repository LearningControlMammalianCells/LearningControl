import pandas as pd
import os
import scipy
from functions import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    folder = "../Experiments"
    exp = 'nd08'
    folder = f'{folder}/{exp}'
    selected_chamber = 1
    format = 't000001xy1c1.tif'

    # print(find_max(folder))
    path_m = f'{folder}/masks'
    if not os.path.isdir(path_m):
        os.mkdir(path_m)

    path_r = f'{folder}/roi'
    if not os.path.isdir(path_r):
        os.mkdir(path_r)

    #path_mat = f'{folder}/{exp}.mat'
    #u = extract_u(path_mat,id = 'Lorena')
    data = pd.read_csv(f'{folder}/log_{exp}.txt',header = None)
    u = data[2]
    print(u)

    precomputed = True
    cropped = True
    Ts = 15
    fluorescence = []
    n_camerette = 5
    Tf = 193
    n_camerette = n_camerette + 1

    if cropped == False:
        for chamber in range(1, n_camerette):
            if chamber is not selected_chamber:
                print(f'work on {chamber}')
                file_name = extract_file_name(folder, format, chamber,t=1)
                roi = cropping(folder,chamber, file_name)
                #diameter = select_diameter(folder,chamber, file_name,roi)


    mean_fluos= []


    for chamber in range(1, n_camerette):
        if chamber != selected_chamber:
            print(chamber)

            roi = np.load(f'{folder}/roi/roi_{chamber}.npy')
            #diameter = int(np.load(f'{folder}/roi/diameter_{chamber}.npy'))
            diameter = 50

            ch1 = extract_paths(folder, format,chamber,Tf )

            if precomputed == False:
                masks = segment(folder, chamber,  ch1, roi, diameter)
            else:
                masks = np.load(f'{folder}/masks/{chamber}.npy')

            fluos = fluo_eval(ch1,masks,roi)
            mean_fluos.append(fluos)
        else:
            mean_fluos.append(list(data[1]))



    #NB CAMBIA DA QUELLI DI LORENA E QUELLI ADELE
    for i in range(5):
        print(mean_fluos[i])
    print(u)
    mean_fluos.append(u)
    df = pd.DataFrame(mean_fluos)
    df = df.T
    df.to_csv(f'{folder}/Dataset_{exp}.csv')
    print('Finito')
