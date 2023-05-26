from functions import *
import os
from deepMPC import deepMPC
from tensorflow import keras
import time
import numpy as np
from scipy.signal import savgol_filter

if __name__ == '__main__':

    # initialization phase
    folder = '../Experiments/nd0fake'
    format_file = 't000001xy1c1.tif'
    selected_chamber = 2
    target = 0.5

    # load network
    model_LSTM = keras.models.load_model('myLSTM')
    n_past = model_LSTM.layers[0].get_output_at(0).get_shape().as_list()[1] - 1
    t = 1

    if t ==1:
        #inizialliza tutto
        with open("MatLabCode/flag.txt", "w+") as f:
            f.write("0")

        if not os.path.isdir(folder):
            print('There is no folder')

        if not os.path.isdir(f'{folder}/masks'):
            os.mkdir(f'{folder}/masks')

        if not os.path.isdir(f'{folder}/masks/{selected_chamber}'):
            os.mkdir(f'{folder}/masks/{selected_chamber}')

        if not os.path.isdir(f'{folder}/roi'):
            os.mkdir(f'{folder}/roi')

        if os.path.isfile(f'{folder}/log.txt'):
            os.remove(f'{folder}/log.txt')

        file_name = extract_file_name(folder, format=format_file, chamber=selected_chamber, t=1)
        roi = cropping(file_name)  # save roi
        np.save(f'{folder}/roi/roi_{selected_chamber}.npy', roi)

        diameter = select_diameter(file_name, roi)
        np.save(f'{folder}/roi/diameter_{selected_chamber}.npy', diameter)

        # vedi fuoco
        #initial_fuoco = lv_fuoco(file_name, roi)

        #cancella img
        for img in os.listdir(folder):
            if 'tif' in img:
                os.remove(f'{folder}/{img}')
        print('Start microscope again')

        states_vector = []
        actions_vector = []
        calibration = []
    else:
        #inizializza solo flag
        with open("MatLabCode/flag.txt", "w+") as f:
            f.write("0")

        states_vector = []
        actions_vector = []
        calibration = []
        with open(f'{folder}/log.txt', "r") as f:
            for row in f:
                _,fluo,a = row.split(',')
                fluo = float(fluo)
                a = float(a)
                states_vector.append(fluo)
                actions_vector.append(a)

        calibration = states_vector[:12]
        roi = np.load(f'{folder}/roi/roi_{selected_chamber}.npy')
        diameter = np.load(f'{folder}/roi/diameter_{selected_chamber}.npy')
        print('NB CONTROLLA LO STATE DELLE SIRINGHE NELLA CARTELLA MATLAB!!!!!')


    #inizializza tempo

    #print('Ora si parte!')
    while 0==0:
        print(f'Ciclo numero {t}')

        file_name = extract_file_name(folder,format=format_file, chamber=selected_chamber, t=t)
        while not os.path.exists(file_name):
            time.sleep(1)
        mask = segment(file_name, roi,diameter)
        np.save(f'{folder}/masks/{selected_chamber}/{t}.npy', mask)

        '''fuoco_t = lv_fuoco(file_name, roi)
        if fuoco_t/initial_fuoco < 90/100:
            print('Perso fuoco')'''


        file_name = file_name.replace('c1', 'c2')
        while not os.path.exists(file_name):
            time.sleep(1)
        fluo = fluo_eval(mask,file_name,roi)
        if np.isnan(fluo):
            print(f'nan value of fluo at time {t}')
            fluo = states_vector[-1]


        states_vector.append(fluo)
        actions_vector.extend([-100]) #nb l'azione è indietro di 1 metti un fill


        if t <=12:
            a = 0 #0: NO TETRACILINA
            calibration.append(fluo)
        else:
            #Control Algorithm
            max_fluo = np.mean(calibration)

            #fai filtro
            states_tmp = savgol_filter(states_vector, 11, 3)
            initial_state = states_tmp[-(n_past + 1):]
            initial_state = initial_state/max_fluo

            initial_action = actions_vector[-(n_past + 1):]

            initial_input = np.array(list(zip(initial_state, initial_action)))
            a = deepMPC(model_LSTM,initial_input,target)


        actions_vector[-1] = a


        #save log file somewhere

        with open(f"{folder}/log.txt", "a") as f:
            f.write(f"{t},{fluo},{a} \n")
            f.close()


        #parla al matlab
        print(f'MATLAB fai {a}')
        #scrivi in flag scrivi in action

        with open("MatLabCode/action.txt", "w+") as f:
            f.write(f"{a}")
            f.close()

        with open("MatLabCode/flag.txt", "w+") as f:
            f.write("1")
            f.close()

        if t>12:
            np.savetxt(f'{folder}/norm_fluo.txt',states_vector/max_fluo)

        t = t + 1
        time.sleep(1)


