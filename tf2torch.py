import tensorflow as tf
import torch
import tf2onnx
import onnx
import onnx2torch
import torchinfo

from components.gesture_dataset import gesture_data
import config as cfg
import numpy as np

import scipy.stats

import sys

model_name="/Users/osamaabdouh/Desktop/NewProfiler/best-model.keras"
tf_model = tf.keras.models.load_model(model_name)
tf_model.save("tf_model.h5")  # Save the model in Keras format

tf_model = tf.keras.models.load_model("tf_model.h5")  # Load the model again to ensure it's in the correct format
onnx_model = tf2onnx.convert.from_keras(tf_model, output_path="model.onnx")
onnx_model = onnx.load("model.onnx")
pytorch_model = onnx2torch.convert(onnx_model)
print(pytorch_model)
# Save the PyTorch model
torch.save(pytorch_model, "model.pth")
# Load the PyTorch model
loaded_model = torch.load("model.pth", weights_only=False)
# Print the loaded model to verify
print(torchinfo.summary(loaded_model))
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
    print("ok3")
    
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
    acc_score = accuracy(output, Y_test)
    
    print(f'Accuracy: {acc_score * 100:.2f}%')