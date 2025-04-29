# Standard library
import os
import math
from collections import Counter

# Data handling & computation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Google Colab
from google.colab import drive

# Scikit-learn
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    auc
)

# SciPy
from scipy import stats
from scipy.stats import multivariate_normal

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Medical imaging
import nibabel as nib
import h5py

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import backend as K, regularizers
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    GlobalAveragePooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
    UpSampling2D,
    Lambda,
    Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import to_categorical


batch_size = 64 #128
epochs =  200 #50 # 100 #200
inChannel = 3
x, y = 256, 256
input_img = Input(shape = (x, y, inChannel))
num_classes = 2
inner_dim = 2048  #1024  #2048 #4096
dropout_rate = 0.5
lr= 0.0001
beta_1 = 0.05

#########################################
# 1. Define Cyclical Learning Rate Callback
#########################################
def cyclical_lr(epoch, base_lr=1e-4, max_lr=1e-3, step_size=5):
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
    return lr

clr_callback = LearningRateScheduler(
    lambda epoch: cyclical_lr(epoch, base_lr=1e-4, max_lr=1e-3, step_size=5),
    verbose=1
)

#########################################
# 2. Variational Autoencoder (VAE) Implementation
#########################################
latent_dim = 128

def vae_encoder(input_img, latent_dim=128):
    """
    Maps the input image to a latent representation.
    Returns:
        z_mean: Mean of the latent distribution.
        z_log_var: Log variance of the latent distribution.
        conv_feature: Intermediate features for later use.
    """
    conv1 = Conv2D(16, (3, 3), padding='same',
                   activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1))(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, (3, 3), strides=(2, 2), padding='same',
                   activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1))(conv1)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(32, (3, 3), padding='same',
                   activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                   activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1))(conv2)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv2D(64, (3, 3), padding='same',
                   activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1))(conv2)
    conv3 = BatchNormalization()(conv3)

    conv_feature = conv3
    flat = Flatten()(conv3)
    z_mean = Dense(latent_dim)(flat)
    z_log_var = Dense(latent_dim)(flat)
    return z_mean, z_log_var, conv_feature

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def vae_decoder(latent_dim):
    """
    Reconstructs an image from the latent vector.
    """
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(4 * 4 * 64, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1))(latent_inputs)
    x = Reshape((4, 4, 64))(x)
    up1 = UpSampling2D((2,2))(x)
    conv4 = Conv2D(32, (3, 3), padding='same',
                   activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1))(up1)
    conv4 = BatchNormalization()(conv4)
    up2 = UpSampling2D((2,2))(conv4)
    conv5 = Conv2D(16, (3, 3), padding='same',
                   activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1))(up2)
    conv5 = BatchNormalization()(conv5)
    decoded = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(conv5)
    return Model(latent_inputs, decoded, name='decoder')

# Build the VAE.
input_img = Input(shape=(16, 16, 3), name="encoder_input")
z_mean, z_log_var, conv_feature = vae_encoder(input_img, latent_dim)
z = Lambda(sampling, name="z")([z_mean, z_log_var])
decoder = vae_decoder(latent_dim)
decoded = decoder(z)
vae = Model(input_img, decoded, name='vae')

# --- Loss Computation Wrapped in a Lambda Layer ---
def total_loss_fn(tensors):
    """
    Compute per-sample VAE loss.
    Returns a tensor of shape (batch, 1).
    """
    inp, dec, z_mean, z_log_var = tensors
    # Compute per-sample reconstruction loss: average over image dimensions.
    rec_loss = tf.reduce_mean(tf.square(inp - dec), axis=[1,2,3]) * (16 * 16 * 3)
    # Compute per-sample KL divergence loss.
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss = -0.5 * kl_loss
    total_loss = rec_loss + kl_loss  # shape: (batch,)
    # Expand dims to ensure output shape is (batch, 1)
    total_loss = tf.expand_dims(total_loss, axis=-1)
    return total_loss

loss_output = Lambda(total_loss_fn, name="total_loss")([input_img, decoded, z_mean, z_log_var])

# Create a dual-output model.
vae_with_loss = Model(
    inputs=input_img,
    outputs={'reconstruction': decoded, 'total_loss': loss_output}
)

# --- Dummy Loss Functions ---
def zero_loss(y_true, y_pred):
    return tf.reduce_mean(0 * y_pred)

def identity_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)

vae_with_loss.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-4),
    loss={'reconstruction': zero_loss, 'total_loss': identity_loss}
)

vae_with_loss.summary()

#########################################
# Data Augmentation and Callbacks
#########################################
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

cp_cb = ModelCheckpoint(
    filepath='../checkpoints/vae_bottleneck_withoutflatten_10x_V3.h5',
    monitor='val_total_loss',
    verbose=1,
    save_best_only=True,
    mode='auto'
)

#########################################
# Training Setup
#########################################
epochs = 100
batch_size = 64

# Prepare targets: for "reconstruction" we want the original image,
# and for "total_loss" we supply dummy zeros.
train_targets = {
    'reconstruction': train_X,
    'total_loss': np.zeros((train_X.shape[0], 1))
}
valid_targets = {
    'reconstruction': valid_X,
    'total_loss': np.zeros((valid_X.shape[0], 1))
}

print("Before fit VAE")
vae_history = vae_with_loss.fit(
    train_X,
    train_targets,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(valid_X, valid_targets),
    shuffle=True,
    callbacks=[cp_cb, clr_callback]
)
print("After fit VAE")

#########################################
# Plot Training History
#########################################
loss_history = vae_history.history['loss']
val_loss_history = vae_history.history['val_loss']
epochs_range = range(len(loss_history))
print("loss", loss_history)
print("val_loss", val_loss_history)

plt.figure()
plt.plot(epochs_range, loss_history, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss_history, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#########################################
# Evaluate and Predict on Test Data
#########################################
test_loss = vae_with_loss.evaluate(
    test_X,
    {'reconstruction': test_X, 'total_loss': np.zeros((test_X.shape[0], 1))},
    verbose=1
)
print("Test Loss:", test_loss)

liverImage_test = vae_with_loss.predict(test_X)['reconstruction']
liverImage_val = vae_with_loss.predict(valid_X)['reconstruction']
print("Test Avg: {0}\nValidation Avg: {1}".format(np.average(liverImage_test), np.average(liverImage_val)))

#########################################
# Visualization
#########################################
def showVersions(orig, dec, num=10):
    plt.figure(figsize=(10, 2 * num))
    for i in range(num):
        plt.subplot(num, 2, 2 * i + 1)
        plt.imshow(orig[i])
        plt.title("Original")
        plt.axis('off')
        plt.subplot(num, 2, 2 * i + 2)
        plt.imshow(dec[i])
        plt.title("Reconstructed")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

showVersions(test_X, liverImage_test, num=10)
print("End of VAE")

#########################################
# 3. Classification Model (FC) with Changed Architecture
#########################################
# Process the labels by reshaping them to 1D arrays.
train_labels = train_labels.reshape(-1)
val_labels = val_labels.reshape(-1)
test_labels = test_labels.reshape(-1)
num_classes = 2  # Set the number of classes for the classification problem.

# Convert integer labels to one-hot encoded vectors.
train_Y_one_hot = to_categorical(train_labels, num_classes=num_classes)
val_Y_one_hot = to_categorical(val_labels, num_classes=num_classes)
test_Y_one_hot = to_categorical(test_labels, num_classes=num_classes)
train_label = train_Y_one_hot
valid_label = val_Y_one_hot

# Define the classification head using transposed convolution and a compact dense layer structure.
def fc(enco):
    """
    Fully-connected classification head.
    Parameters:
        enco: Convolutional feature maps (output from encoder).
    Returns:
        x: Final output tensor with softmax activation for class probabilities.
    """
    # Use a transposed convolution to upsample the features.
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',
                          activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1))(enco)
    x = BatchNormalization()(x)
    # Flatten the feature maps into a single vector.
    flat = Flatten()(x)
    # First compact dense layer with 512 units.
    x = Dense(512, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
              kernel_regularizer=regularizers.l2(1e-4))(flat)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    # Second compact dense layer with 128 units.
    x = Dense(128, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
              kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    # Final dense layer with softmax activation to output class probabilities.
    out = Dense(num_classes, activation='softmax',
                kernel_regularizer=regularizers.l2(1e-4))(x)
    return out

# Build the classifier model using the convolutional features (conv_feature) from the VAE encoder.
# (Assuming that conv_feature and input_img are defined by your previously-built VAE model.)
classifier_output = fc(conv_feature)
full_model = Model(input_img, classifier_output)

# Optionally, copy weights from the VAE encoder to the corresponding layers in the classifier.
for l1, l2 in zip(full_model.layers[:len(vae.layers)], vae.layers[:len(full_model.layers)]):
    weights_l1 = l1.get_weights()
    weights_l2 = l2.get_weights()
    if len(weights_l1) == len(weights_l2):
        l1.set_weights(weights_l2)
    else:
        print(f"Skipping layer {l1.name}: expected {len(weights_l1)} weights but got {len(weights_l2)}")

# Ensure that the encoder layers (the first part of the network) remain trainable.
for layer in full_model.layers[:len(vae.layers)]:
    layer.trainable = True

# Compile the classifier model using AdamW with weight decay.
classifier_optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-4)
full_model.compile(loss='categorical_crossentropy', optimizer=classifier_optimizer, metrics=['accuracy'])

# Set up callbacks for classifier training, including checkpointing.
cp_cb_classifier = ModelCheckpoint(filepath='../checkpoints/vae_classification_10x_V3.h5',
                                   monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [cp_cb_classifier, clr_callback]

# Print the full classifier model summary.
full_model.summary()
print("End of FC initialization")

# Train the classifier model.
classify_train = full_model.fit(train_X, train_label,
                                batch_size=64,
                                epochs=100,
                                verbose=1,
                                callbacks=callbacks_list,
                                validation_data=(valid_X, valid_label),
                                shuffle=True)

# Retrieve training history metrics.
accuracy_hist = classify_train.history.get('accuracy', classify_train.history.get('acc'))
val_accuracy_hist = classify_train.history.get('val_accuracy', classify_train.history.get('val_acc'))
loss_train = classify_train.history['loss']
val_loss_train = classify_train.history['val_loss']
epochs_range = range(len(accuracy_hist))

# Plot the training and validation accuracy over epochs.
plt.figure()
plt.plot(epochs_range, accuracy_hist, 'bo', label='Training accuracy')
plt.plot(epochs_range, val_accuracy_hist, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

# Plot the training and validation loss over epochs.
plt.figure()
plt.plot(epochs_range, loss_train, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss_train, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Evaluate the classifier on the test dataset.
test_eval = full_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# Predict class probabilities on the test set.
predicted_probs = full_model.predict(test_X)
# Convert probabilities to discrete class labels.
predicted_classes = np.argmax(np.round(predicted_probs), axis=1)
correct = np.where(predicted_classes == test_labels)[0]
incorrect = np.where(predicted_classes != test_labels)[0]
print("Found %d correct labels" % len(correct))
print("Found %d incorrect labels" % len(incorrect))

# Generate a detailed classification report.
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_labels, predicted_classes, target_names=target_names))
# Compute confusion matrix.
cm = confusion_matrix(test_labels, predicted_classes)
TN, FP, FN, TP = cm.ravel()
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print("Sensitivity (Recall for positive class): {:.2f}".format(sensitivity))
print("Specificity (Recall for negative class): {:.2f}".format(specificity))

#########################################
# ROC Curve Computation and Plotting
#########################################
# For ROC curve, extract the probability for the positive class (assuming index 1).
# predicted_probs has shape (num_samples, num_classes). We need the column for the positive class.
y_scores = predicted_probs[:, 1]

# Compute ROC curve and AUC.
fpr, tpr, thresholds = roc_curve(test_labels, y_scores)
roc_auc = auc(fpr, tpr)

# Specificity values are 1 - FPR.
spec_vals = 1 - fpr

# Compute Youden’s Index (TPR - FPR) to determine the optimal threshold.
youden_index = tpr - fpr
best_index = np.argmax(youden_index)
best_threshold = thresholds[best_index]
best_sensitivity = tpr[best_index]
best_specificity = spec_vals[best_index]

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Annotate the best threshold.
plt.plot(fpr[best_index], tpr[best_index], 'ro', markersize=8)
plt.text(fpr[best_index] + 0.02, tpr[best_index] - 0.05,
         f'Threshold: {best_threshold:.2f}\nSensitivity: {best_sensitivity:.2f}\nSpecificity: {best_specificity:.2f}',
         fontsize=10, color='red')
plt.show()
