import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from src.net import mlp_random, cnn_random
from src.hyperparameters import get_hyperparameters_mlp, get_hyperparemeters_cnn
from src.utils import load_ctf_2025, AES_Sbox, evaluate, calculate_HW

if __name__ == "__main__":
    root = './'
    dataset = "CHES_2025"
    model_type = "cnn" #mlp, cnn
    leakage = "HW" #ID, HW
    train_models = True
    epochs = 50

    seed = 0
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    nb_traces_attacks = 1700
    total_nb_traces_attacks = 2000


    if not os.path.exists('./Result/'):
        os.mkdir('./Result/')

    root = "./Result/"
    save_root = root+dataset+"_"+model_type+"_"+leakage+"/"
    model_root = save_root+"models/"
    print("root:", root)
    print("save_time_path:", save_root)
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    if not os.path.exists(model_root):
        os.mkdir(model_root)

    if dataset == "CHES_2025":
        byte = 2
        data_root = './../Dataset/CHES_2025/CHES_Challenge_v0.h5' #change this to where you download your dataset.
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ctf_2025(
            data_root, leakage_model="ID", byte=byte, train_begin=0, train_end=45000, test_begin=0,
            test_end=10000)

    if leakage == 'ID':
        def leakage_fn(att_plt, k):
            return AES_Sbox[k ^ int(att_plt)]
        classes = 256
    elif leakage == 'HW':
        def leakage_fn(att_plt, k):
            hw = [bin(x).count("1") for x in range(256)]
            return hw[AES_Sbox[k ^ int(att_plt)]]
        classes = 9
        Y_profiling = calculate_HW(Y_profiling)
        Y_attack = calculate_HW(Y_attack)

    scaler_std = StandardScaler()
    X_profiling = scaler_std.fit_transform(X_profiling)
    X_attack = scaler_std.transform(X_attack)
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
    number_of_samples = X_profiling.shape[1]
    X_attack_test, X_attack_val, Y_attack_test, Y_attack_val = train_test_split(X_attack,Y_attack,test_size=0.1,
                                                                                                    random_state=0)

    if train_models == True:
        regularization = False
        if model_type == "mlp":
            hp = get_hyperparameters_mlp(regularization=regularization)
            model, seed, hp = mlp_random(classes, number_of_samples, regularization=regularization, hp=hp)
        else:
            hp = get_hyperparemeters_cnn(regularization=regularization)
            model, seed, hp = cnn_random(classes, number_of_samples, regularization=regularization, hp=hp)
        hp["epochs"] = epochs
        model.fit(x=X_profiling, y=to_categorical(Y_profiling, num_classes=classes), batch_size=hp["batch_size"], verbose=2,
                      epochs=hp["epochs"],  validation_data=(X_attack_val, to_categorical(Y_attack_val, num_classes=classes)))
        model.save(save_root + "my_model_0.h5")
    else:
        model = tf.keras.models.load_model(save_root +'my_model_0.h5')

    GE, NTGE = evaluate(model, X_attack, plt_attack, correct_key, leakage_fn=leakage_fn, nb_attacks=100,
                        total_nb_traces_attacks=2000, nb_traces_attacks=100)