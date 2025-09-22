import datetime
import sys
import os

os.system("")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import tensorflow as tf
import tf2onnx
import onnx
import onnx2torch
import torch
import torchinfo

from components.colors import colors
from components.dataset import cifar_data, mnist
from components.gesture_dataset import gesture_data

import config as cfg
import numpy as np

import scipy.stats


import questionary


def check_model_path(path):
    if os.path.isfile(path):
        return True
    else:
        print(colors.FAIL, f"{path}: is not a file or does not exist.", colors.ENDC)
    
import sys
import os

def call_silently(func, *args, **kwargs):
    """Executes a function while temporarily disabling stdout."""
    saved_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')  # Disable output
    try:
        return func(*args, **kwargs)
    finally:
        sys.stdout.close()
        sys.stdout = saved_stdout  # Restores output

def load_trained_model(m_path):
    # Ask the user if they want to see the user parameters configuration and model summary
    flag=questionary.confirm("Want to see User params configuration and Model summary?", default=False).ask()

    # FOLDER SECTION --------------------------------------------------------------------------------------------------------

    # list of folders that can be added to the tuner folder if missing
    new_dir = ['Model', 'Weights', 'database', 'checkpoints', 'log_folder', 'algorithm_logs', 'dashboard/model', 'symbolic']

    if flag:
        # cfg.get_experiment_name()
        print(colors.MAGENTA, "|  ----------- USER PARAMS CONFIGURATION ----------  |\n", colors.ENDC)
        print("EXPERIMENT NAME: ", cfg.NAME_EXP)
        print("DATASET NAME: ", cfg.DATA_NAME)
        print("MAX NET EVAL: ", cfg.MAX_EVAL)
        print("EPOCHS FOR TRAINING: ", cfg.EPOCHS)
        print("MODULE LIST: ", cfg.MOD_LIST)
        # iterate over each name in the list of folders and
        # if it doesn't exist, proceed with its creation
        
    try:
        if not os.path.exists("{}".format(cfg.NAME_EXP)):
            os.makedirs(cfg.NAME_EXP)
    except OSError:
        print(colors.FAIL, "|  ----------- FAILED TO CREATE FOLDER ----------  |\n", colors.ENDC)
        exit()
    for folder in new_dir:
        try:
            if not os.path.exists("{}/{}".format(cfg.NAME_EXP,folder)):
                os.makedirs("{}/{}".format(cfg.NAME_EXP,folder))
        except OSError:
            print(colors.FAIL, "|  ----------- FAILED TO CREATE FOLDER {} ----------  |\n".format("{}/{}".format(cfg.NAME_EXP,folder)), colors.ENDC)
            exit()
    try:
        os.system("cp symbolic_base/* {}/symbolic/".format(cfg.NAME_EXP))
    except OSError:
        print(colors.FAIL, "|  ----------- FAILED TO COPY SYMBOLIC DIR ----------  |\n".format("{}/{}".format(cfg.NAME_EXP,folder)), colors.ENDC)
        exit()
        1
    if cfg.DATA_NAME == "MNIST":
        # MNIST SECTION --------------------------------------------------------------------------------------------------------
        if flag:
            X_train, X_test, Y_train, Y_test, n_classes = mnist()
        else:
            X_train, X_test, Y_train, Y_test, n_classes = call_silently(mnist)

    elif cfg.DATA_NAME == "CIFAR-10":
        # CIFAR-10 SECTION -----------------------------------------------------------------------------------------------------
        # obtain images and labels from the cifar dataset
        if flag:
            X_train, X_test, Y_train, Y_test, n_classes = cifar_data()
        else:
            X_train, X_test, Y_train, Y_test, n_classes = call_silently(cifar_data)
    elif cfg.DATA_NAME == "gesture":
        # GestureDVS128 SECTION -----------------------------------------------------------------------------------------------------
        if flag:
            X_train, X_test, Y_train, Y_test, n_classes = gesture_data()
        else:
            X_train, X_test, Y_train, Y_test, n_classes = call_silently(gesture_data)
    else:
        print(colors.FAIL, "|  ----------- DATASET NOT FOUND ----------  |\n", colors.ENDC)
        sys.exit()
        
    dt = datetime.datetime.now()
    max_evals = cfg.MAX_EVAL

    # LOADING ALREADY TRAINED MODEL --------------------------------------------------------------------------------------------------------
    
    check_model_path(m_path)

    try:
        tf_model = tf.keras.models.load_model(m_path)
        tf_model.save("tf_model.h5")  # Save the model in Keras format

        tf_model = tf.keras.models.load_model("tf_model.h5")  # Load the model again to ensure it's in the correct format
        onnx_model = tf2onnx.convert.from_keras(tf_model, output_path="model.onnx")
        onnx_model = onnx.load("model.onnx")
        pytorch_model = onnx2torch.convert(onnx_model)
        print(pytorch_model)
        #save the PyTorch model
        torch.save(pytorch_model, "model.pth")
        # Load the PyTorch model
        loaded_model=torch.load("model.pth", weights_only=False)
        print(torchinfo.summary)
    except Exception as e:
        print(colors.FAIL, f"Error loading PyTorch model: {e}", colors.ENDC)
        return None
    print("conversione modello in pytorch")
    cfg.MODE = "fwdPass"
    cfg.NUM_CHANNELS = 2
    cfg.FRAMES = 16
    cfg.POLARITY = "both"

    try:
        print("Caricamento del dataset...")
        X_train, X_test, Y_train, Y_test, n_classes = gesture_data()
        print("Dataset caricato con successo!")
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    def eval_net(model, X, y):
        val_outputs_list = []
        loss_fn = torch.nn.CrossEntropyLoss()
        for t in range(X.shape[1]):
            print(f"Shape of X[:, {t}]: {X[:, t].shape}")
            val_frame_outputs = model(X[:, t])
            val_outputs_list.append(val_frame_outputs)

        val_outputs = torch.stack(val_outputs_list, axis=1)
        val_outputs = torch.tensor(val_outputs, dtype=torch.float32)
        return val_outputs

    def accuracy(outputs, targets):
        """
        Computes classification accuracy by selecting the most frequent predicted class.
        """
        output_frames = tf.argmax(outputs, axis=2).numpy()
        mode, _ = scipy.stats.mode(output_frames, axis=1, keepdims=False)
        output_preds = tf.cast(mode, tf.int64)

        target_frames = tf.argmax(targets, axis=2).numpy()
        target_mode, _ = scipy.stats.mode(target_frames, axis=1, keepdims=False)

        return tf.reduce_mean(tf.cast(output_preds == target_mode, tf.float64)).numpy()


    with torch.no_grad():
        loaded_model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.long)
        # Forward pass through the model
        output = eval_net(loaded_model, X_test, Y_test)
        
        # Print the output shape
        print("Output shape:", output.shape)
        print("Y_test shape:", Y_test.shape)
        
        # If you want to compute accuracy or any other metric, you can do it here
        accuracy = accuracy(output, Y_test)
        
        print(f'Accuracy: {accuracy * 100:.2f}%')

    if flag:
        load_model.summary()
        
    return load_model