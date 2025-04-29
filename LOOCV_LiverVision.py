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

# For reproducibility.
np.random.seed(42)
tf.random.set_seed(42)

#########################################
# 1. Data Loading: Read 6 patients (IDs 0 to 5) and extract liver slices
#########################################
data_dir = "/content/drive/My Drive/Liver"
patient_data = []

for patient_id in range(6):
    ct_filename = f"volume-{patient_id}.nii"
    seg_filename = f"segmentation-{patient_id}.nii"

    ct_path = os.path.join(data_dir, ct_filename)
    seg_path = os.path.join(data_dir, seg_filename)

    if not (os.path.exists(ct_path) and os.path.exists(seg_path)):
        print(f"[WARNING] CT or segmentation file not found for patient {patient_id}. Skipping...")
        continue

    print(f"\nLoading patient {patient_id} data...")
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()

    mask_img = nib.load(seg_path)
    mask_data = mask_img.get_fdata()

    print(f"Patient {patient_id}: CT data shape: {ct_data.shape}")
    print(f"Patient {patient_id}: Segmentation data shape: {mask_data.shape}")

    # Check that we have the expected labels: {0, 1, 2}
    unique_labels = np.unique(mask_data)
    if set(unique_labels) != {0, 1, 2}:
        print(f"[WARNING] Patient {patient_id}: Expected labels {{0,1,2}} but found {set(unique_labels)}. Skipping patient.")
        continue

    # Identify slices containing liver (mask value 1)
    liver_slices = [s for s in range(mask_data.shape[2]) if np.any(mask_data[:, :, s] == 1)]
    if not liver_slices:
        print(f"Patient {patient_id}: No liver (label 1) found. Skipping patient.")
        continue

    print(f"Patient {patient_id}: Liver found in slices: {liver_slices}")

    # Optional visualization of liver slices.
    num_slices = len(liver_slices)
    cols = 5
    rows = math.ceil(num_slices / cols)
    plt.figure(figsize=(cols * 3, rows * 3))
    plt.suptitle(f"Patient {patient_id} - Liver Mask Slices", fontsize=16)
    for i, slice_idx in enumerate(liver_slices):
        plt.subplot(rows, cols, i+1)
        plt.imshow(mask_data[:, :, slice_idx], cmap='gray')
        plt.title(f"Slice {slice_idx}")
        plt.axis("off")
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    ct_volume_with_liver = ct_data[:, :, liver_slices]
    mask_volume_with_liver = mask_data[:, :, liver_slices]
    patient_data.append((patient_id, ct_volume_with_liver, mask_volume_with_liver))

print("\nData loading and liver slice extraction completed.")
print(f"Total patients processed: {len(patient_data)}")

#########################################
# 2. Define 6-Fold Cross Validation Splits
#########################################
# For each fold, use:
# - Test: patient i
# - Validation: patient (i+1) mod 6
# - Training: all others (4 patients)
folds = []
num_patients = len(patient_data)
for i in range(num_patients):
    test_idx = i
    valid_idx = (i + 1) % num_patients
    train_idxs = [j for j in range(num_patients) if j not in [test_idx, valid_idx]]
    folds.append({"train": train_idxs, "valid": [valid_idx], "test": [test_idx]})

#########################################
# 3. Helper Functions: Stratification, Patch Extraction, and RGB Conversion
#########################################
def stratify_patient_data(split_data):
    max_slices = max(ct_vol.shape[2] for _, ct_vol, _ in split_data)
    stratified_split = []
    for patient_id, ct_vol, mask_vol in split_data:
        current_slices = ct_vol.shape[2]
        if current_slices == max_slices:
            stratified_split.append((patient_id, ct_vol, mask_vol))
        else:
            new_indices = np.linspace(0, current_slices - 1, num=max_slices)
            new_indices = np.round(new_indices).astype(int)
            new_indices[new_indices >= current_slices] = current_slices - 1
            new_ct_vol = ct_vol[:, :, new_indices]
            new_mask_vol = mask_vol[:, :, new_indices]
            print(f"Patient {patient_id} stratified from {current_slices} to {new_ct_vol.shape[2]} slices.")
            stratified_split.append((patient_id, new_ct_vol, new_mask_vol))
    return stratified_split

def extract_patches_with_labels(patient_split, patch_size=16):
    """
    For each patch:
      - patch_ct: 16x16 CT patch.
      - patch_mask: 16x16 segmentation patch with labels (0: background, 1: liver, 2: tumor).
      - label: binary label = 1 if any pixel in patch_mask is 2, 0 otherwise.
    Returns numpy arrays: ct_patches, mask_patches, patch_labels.
    """
    ct_patches = []
    mask_patches = []
    patch_labels = []

    for patient_id, ct_vol, mask_vol in patient_split:
        num_slices = ct_vol.shape[2]
        print(f"Extracting patches for patient {patient_id} with {num_slices} slices...")
        for s in range(num_slices):
            slice_ct = ct_vol[:, :, s]
            slice_mask = mask_vol[:, :, s]
            for i in range(0, slice_ct.shape[0], patch_size):
                for j in range(0, slice_ct.shape[1], patch_size):
                    patch_ct = slice_ct[i:i+patch_size, j:j+patch_size]
                    patch_mask = slice_mask[i:i+patch_size, j:j+patch_size]
                    if patch_ct.shape[0] == patch_size and patch_ct.shape[1] == patch_size:
                        label = 1 if np.any(patch_mask == 2) else 0
                        ct_patches.append(patch_ct)
                        mask_patches.append(patch_mask)
                        patch_labels.append(label)
    return np.array(ct_patches), np.array(mask_patches), np.array(patch_labels)

def convert_to_rgb(ct_patches):
    if ct_patches.ndim == 3:
        ct_patches = np.expand_dims(ct_patches, axis=-1)
    return np.repeat(ct_patches, 3, axis=-1)

#########################################
# 4. Model Components & Callbacks for VAE and Classifier
#########################################
# Cyclical Learning Rate Callback.
def cyclical_lr(epoch, base_lr=1e-4, max_lr=1e-3, step_size=5):
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
    return lr

clr_callback = LearningRateScheduler(
    lambda epoch: cyclical_lr(epoch, base_lr=1e-4, max_lr=1e-3, step_size=5),
    verbose=1
)

# VAE components.
latent_dim = 128

def vae_encoder(input_img, latent_dim=128):
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

def total_loss_fn(tensors):
    inp, dec, z_mean, z_log_var = tensors
    rec_loss = tf.reduce_mean(tf.square(inp - dec), axis=[1,2,3]) * (16 * 16 * 3)
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss = -0.5 * kl_loss
    total_loss = rec_loss + kl_loss
    total_loss = tf.expand_dims(total_loss, axis=-1)
    return total_loss

# Dummy loss functions.
def zero_loss(y_true, y_pred):
    return tf.reduce_mean(0 * y_pred)

def identity_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)

# Classification FC head.
def fc(enco):
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',
                          activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1))(enco)
    x = BatchNormalization()(x)
    flat = Flatten()(x)
    x = Dense(512, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
              kernel_regularizer=regularizers.l2(1e-4))(flat)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
              kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    out = Dense(2, activation='softmax',
                kernel_regularizer=regularizers.l2(1e-4))(x)
    return out

def dice_coef(y_true, y_pred, smooth=1):
    # we only care about the “tumor” channel (index 1)
    y_true_f = K.flatten(y_true[..., 1])
    # round the predicted tumor‐probabilities to 0 or 1
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#########################################
# 5. Cross Validation Loop: 6 Folds (Each patient is test once)
#########################################
all_fold_roc = []  # To store ROC data from each fold.

for fold_idx, fold in enumerate(folds):
    print(f"\n========== Starting Fold {fold_idx+1} ==========")
    # Assign patients for this fold.
    train_patients = [patient_data[i] for i in fold["train"]]
    valid_patient = [patient_data[i] for i in fold["valid"]]
    test_patient  = [patient_data[i] for i in fold["test"]]
    print(f"Fold {fold_idx+1} split: {len(train_patients)} train, {len(valid_patient)} valid, {len(test_patient)} test.")

    # Stratify each split.
    train_stratified = stratify_patient_data(train_patients)
    valid_stratified = stratify_patient_data(valid_patient)
    test_stratified  = stratify_patient_data(test_patient)

    # Extract patches.
    train_ct_patches, train_mask_patches, train_patch_labels = extract_patches_with_labels(train_stratified, patch_size=16)
    valid_ct_patches, valid_mask_patches, valid_patch_labels = extract_patches_with_labels(valid_stratified, patch_size=16)
    test_ct_patches,  test_mask_patches,  test_patch_labels  = extract_patches_with_labels(test_stratified, patch_size=16)

    print(f"Fold {fold_idx+1} patch shapes:")
    print("  Train CT:", train_ct_patches.shape, "Train labels:", train_patch_labels.shape)
    print("  Valid CT:", valid_ct_patches.shape, "Valid labels:", valid_patch_labels.shape)
    print("  Test CT:", test_ct_patches.shape, "Test labels:", test_patch_labels.shape)

    # Convert CT patches to RGB.
    train_X = convert_to_rgb(train_ct_patches)
    valid_X = convert_to_rgb(valid_ct_patches)
    test_X  = convert_to_rgb(test_ct_patches)

    # One-hot encode labels.
    train_Y_one_hot = to_categorical(train_patch_labels, num_classes=2)
    valid_Y_one_hot = to_categorical(valid_patch_labels, num_classes=2)
    test_Y_one_hot  = to_categorical(test_patch_labels, num_classes=2)

    #########################################
    # VAE Training.
    #########################################
    encoder_input = Input(shape=(16,16,3), name="encoder_input")
    z_mean, z_log_var, conv_feature = vae_encoder(encoder_input, latent_dim)
    z = Lambda(sampling, name="z")([z_mean, z_log_var])
    decoder = vae_decoder(latent_dim)
    decoded = decoder(z)
    vae = Model(encoder_input, decoded, name='vae')

    loss_output = Lambda(total_loss_fn, name="total_loss")([encoder_input, decoded, z_mean, z_log_var])
    vae_with_loss = Model(inputs=encoder_input, outputs={'reconstruction': decoded, 'total_loss': loss_output})
    vae_with_loss.compile(optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-4),
                          loss={'reconstruction': zero_loss, 'total_loss': identity_loss})
    print("\nBefore VAE training for Fold", fold_idx+1)
    vae_history = vae_with_loss.fit(
        train_X,
        {'reconstruction': train_X, 'total_loss': np.zeros((train_X.shape[0], 1))},
        epochs=10, batch_size=64,
        validation_data=(valid_X, {'reconstruction': valid_X, 'total_loss': np.zeros((valid_X.shape[0], 1))}),
        shuffle=True, callbacks=[clr_callback]
    )
    print("After VAE training for Fold", fold_idx+1)

    #########################################
    # Classifier Training using Encoder Features.
    #########################################
    classifier_output = fc(conv_feature)
    full_model = Model(encoder_input, classifier_output)

    # Optionally copy weights from VAE encoder layers.
    for l1, l2 in zip(full_model.layers[:len(vae.layers)], vae.layers[:len(full_model.layers)]):
        if len(l1.get_weights()) == len(l2.get_weights()):
            l1.set_weights(l2.get_weights())

    for layer in full_model.layers[:len(vae.layers)]:
        layer.trainable = True

    full_model.compile(loss='categorical_crossentropy', optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-4), metrics=['accuracy', dice_coef])
    full_model.summary()

    cp_cb_classifier = ModelCheckpoint(filepath=f'../checkpoints/vae_classification_fold{fold_idx+1}.h5',
                                       monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [cp_cb_classifier, clr_callback]

    classify_train = full_model.fit(train_X, train_Y_one_hot, batch_size=64, epochs=10,
                                    validation_data=(valid_X, valid_Y_one_hot),
                                    shuffle=True, callbacks=callbacks_list)

    # Evaluate the classifier.
    test_eval = full_model.evaluate(test_X, test_Y_one_hot, verbose=0)
    test_eval = full_model.evaluate(test_X, test_Y_one_hot, verbose=0)
    # test_eval = [loss, accuracy, dice_coef]
    print(
        f"Fold {fold_idx+1}  "
        f"Test loss: {test_eval[0]:.4f}  "
        f"Test accuracy: {test_eval[1]:.4f}  "
        f"Test dice: {test_eval[2]:.4f}"
    )

    predicted_probs = full_model.predict(test_X)
    predicted_classes = np.argmax(np.round(predicted_probs), axis=1)

    # Compute ROC for this fold.
    y_scores = predicted_probs[:, 1]  # probability for tumor class.
    fpr, tpr, thresholds = roc_curve(test_patch_labels.reshape(-1), y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"Fold {fold_idx+1} ROC AUC: {roc_auc:.2f}")

    all_fold_roc.append((fpr, tpr, roc_auc))

    print(classification_report(test_patch_labels.reshape(-1), predicted_classes))

#########################################
# 6. After Cross Validation: Plot All 6 ROC Curves Together
#########################################
plt.figure(figsize=(8, 6))
colors = ['b', 'g', 'r', 'c', 'm', 'y']
for i, (fpr, tpr, roc_auc) in enumerate(all_fold_roc):
    plt.plot(fpr, tpr, color=colors[i % len(colors)], label=f'Fold {i+1} ROC (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--', label='Chance')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves for 6-Fold Cross Validation')
plt.legend(loc="lower right")
plt.show()
