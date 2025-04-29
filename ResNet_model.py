import os
import math
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import h5py

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose, Lambda, BatchNormalization,
                                     Flatten, Dense, Reshape, UpSampling2D, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Dense, Dropout, Resizing)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

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

def dice_coef(y_true, y_pred, smooth=1):
    # focus on the “tumor” channel (index 1)
    y_true_f = K.flatten(y_true[..., 1])
    # binarize the predictions
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
    # Convert integer labels to one-hot encoded vectors.
    #########################################
    num_classes = 2
    train_Y_one_hot = to_categorical(train_patch_labels, num_classes=num_classes)
    valid_Y_one_hot = to_categorical(valid_patch_labels, num_classes=num_classes)
    test_Y_one_hot  = to_categorical(test_patch_labels, num_classes=num_classes)

    #########################################
    # Define Cyclical Learning Rate Callback (optional)
    #########################################
    def cyclical_lr(epoch, base_lr=1e-4, max_lr=1e-3, step_size=5):
        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
        return lr

    clr_callback = LearningRateScheduler(
        lambda epoch: cyclical_lr(epoch, base_lr=1e-4, max_lr=1e-3, step_size=5), verbose=1
    )

    #########################################
    # Build a ResNet-based Classifier
    #########################################
    # Input images are 16x16x3; add a resizing layer to scale them to 32x32x3.
    input_res = Input(shape=(16, 16, 3), name='input_res')
    # Resize to 32x32 (ResNet50 requires at least 32x32).
    x = Resizing(32, 32)(input_res)

    # Load a pre-trained ResNet50 base (without the classification head) using imagenet weights.
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(32, 32, 3)
    )

    # Option: Freeze the base model to train only the top layers.
    base_model.trainable = False

    # Pass the resized input through the base ResNet model.
    x = base_model(x)
    # Add global average pooling to reduce the spatial dimensions.
    x = GlobalAveragePooling2D()(x)
    # (Optional) Add a dense layer for further feature extraction.
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    # Final classification layer with softmax activation for 2 classes.
    output_res = Dense(num_classes, activation='softmax')(x)

    # Create the ResNet-based classifier model.
    resnet_model = Model(inputs=input_res, outputs=output_res, name='ResNet_Classifier')
    resnet_model.summary()

    #########################################
    # Compile and Train the ResNet Model
    #########################################
    resnet_optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-4)
    resnet_model.compile(
        optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', dice_coef]
    )

    # Set up a ModelCheckpoint callback.
    cp_cb_resnet = ModelCheckpoint(
        filepath='../checkpoints/resnet_classifier.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='auto'
    )

    epochs = 5
    batch_size = 64

    # Train the ResNet classifier.
    resnet_history = resnet_model.fit(
        train_X, train_Y_one_hot,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(valid_X, valid_Y_one_hot),
        shuffle=True,
        callbacks=[cp_cb_resnet, clr_callback]
    )

    #########################################
    # Evaluate the ResNet Classifier
    #########################################
    test_loss, test_accuracy, test_dice = resnet_model.evaluate(test_X, test_Y_one_hot, verbose=1)
    print(f"ResNet Test Loss: {test_loss:.4f}")
    print(f"ResNet Test Accuracy: {test_accuracy:.4f}")
    print(f"ResNet Test Dice: {test_dice:.4f}")

    # Obtain predictions and convert to class labels.
    predicted_classes = np.argmax(resnet_model.predict(test_X), axis=1)
    print(classification_report(test_patch_labels, predicted_classes))

    # Compute confusion matrix.
    cm = confusion_matrix(test_patch_labels, predicted_classes)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN)  # True Positive Rate.
    specificity = TN / (TN + FP)  # True Negative Rate.
    print("Sensitivity (Recall for positive class): {:.2f}".format(sensitivity))
    print("Specificity (Recall for negative class): {:.2f}".format(specificity))

    #########################################
    # Plot ROC Curve with Sensitivity & Specificity
    #########################################
    # Obtain prediction probabilities from the model.
    # For binary classification, we extract the probability for the positive class.
    y_pred_probs = resnet_model.predict(test_X)
    y_scores = y_pred_probs[:, 1]  # probability for class '1'

    # Compute the ROC curve and AUC.
    fpr, tpr, thresholds = roc_curve(test_patch_labels, y_scores)
    roc_auc = auc(fpr, tpr)
    # … after roc_auc = auc(fpr, tpr)
    all_fold_roc.append((fpr, tpr, roc_auc))

    # Specificity is 1 - fpr.
    specificity_vals = 1 - fpr

    # Calculate Youden’s Index to determine the best threshold.
    youden_index = tpr - fpr
    best_index = np.argmax(youden_index)
    best_threshold = thresholds[best_index]
    best_sensitivity = tpr[best_index]
    best_specificity = specificity_vals[best_index]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Annotate the best threshold point.
    plt.plot(fpr[best_index], tpr[best_index], 'ro', markersize=8)
    plt.text(fpr[best_index] + 0.02, tpr[best_index] - 0.05,
            f'Threshold: {best_threshold:.2f}\nSensitivity: {best_sensitivity:.2f}\nSpecificity: {best_specificity:.2f}',
            fontsize=10, color='red')
    plt.show()

    #########################################
    # Plot Training History for ResNet
    #########################################
    plt.figure()
    plt.plot(resnet_history.history['loss'], 'bo', label='Training Loss')
    plt.plot(resnet_history.history['val_loss'], 'b', label='Validation Loss')
    plt.title('ResNet: Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(resnet_history.history['accuracy'], 'bo', label='Training Accuracy')
    plt.plot(resnet_history.history['val_accuracy'], 'b', label='Validation Accuracy')
    plt.title('ResNet: Training and Validation Accuracy')
    plt.legend()
    plt.show()
plt.figure(figsize=(8, 6))
for idx, (fpr, tpr, roc_auc) in enumerate(all_fold_roc):
    plt.plot(fpr, tpr, label=f'Fold {idx+1} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Aggregated ROC Curves Across 6 Folds')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
