!pip install segmentation-models

import random
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
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
import keras
keras.utils.generic_utils = keras.utils  # Patch for compatibility.
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    UpSampling2D, concatenate, GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.models import Model

# For reproducibility.
np.random.seed(42)
tf.random.set_seed(42)

class PatchSequence(Sequence):
    def __init__(self,
                 patient_splits,
                 batch_size=16,
                 patch_size=16,
                 target_size=(128,128),
                 shuffle=True):
        """
        patient_splits: list of (pid, ct_vol, mask_vol)
        """
        self.batch_size  = batch_size
        self.patch_size  = patch_size
        self.target_size = target_size
        self.shuffle     = shuffle

        # Unpack CT and mask volumes
        self.ct_vols   = [ct for _, ct, _ in patient_splits]
        self.mask_vols = [msk for _, _, msk in patient_splits]

        # Precompute all valid patch locations + labels
        self.locations = []
        for vidx, (ct, msk) in enumerate(zip(self.ct_vols, self.mask_vols)):
            H, W, D = ct.shape
            for s in range(D):
                for i in range(0, H, patch_size):
                    for j in range(0, W, patch_size):
                        if i+patch_size <= H and j+patch_size <= W:
                            patch_mask = msk[i:i+patch_size, j:j+patch_size, s]
                            label = 1 if np.any(patch_mask==2) else 0
                            self.locations.append((vidx, s, i, j, label))

        self.on_epoch_end()

    def __len__(self):
        return len(self.locations) // self.batch_size

    def __getitem__(self, idx):
        batch = self.locations[
            idx*self.batch_size:(idx+1)*self.batch_size
        ]
        X = np.zeros((self.batch_size, *self.target_size, 3), dtype=np.float32)
        y = np.zeros((self.batch_size,), dtype=np.int32)

        for b, (vidx, s, i, j, label) in enumerate(batch):
            patch = self.ct_vols[vidx][i:i+self.patch_size,
                                       j:j+self.patch_size, s]
            # expand to 3‑channel
            patch_rgb = np.stack([patch]*3, axis=-1)
            # resize to model input
            X[b] = cv2.resize(patch_rgb, self.target_size)
            y[b] = label

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.locations)

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
    # convert sparse integer labels → one-hot
    y_true_oh = K.one_hot(K.cast(y_true, 'int32'), 2)
    # pull out the “tumor” class channel
    y_true_f = K.flatten(y_true_oh[..., 1])
    y_pred_f = K.flatten(y_pred[..., 1])
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
    train_seq = PatchSequence(train_stratified,
                              batch_size=16,
                              patch_size=16,
                              target_size=(128,128),
                              shuffle=True)
    valid_seq = PatchSequence(valid_stratified,
                              batch_size=16,
                              patch_size=16,
                              target_size=(128,128),
                              shuffle=False)
    test_seq  = PatchSequence(test_stratified,
                              batch_size=16,
                              patch_size=16,
                              target_size=(128,128),
                              shuffle=False)

    #########################################
    # Convert integer labels to one-hot encoded vectors.
    #########################################
    num_classes = 2

    ############################################
    # Set segmentation_models framework and preprocessing.
    ############################################

    def conv_block(x, filters, kernel_size=(3,3), padding='same'):
        x = Conv2D(filters, kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def build_manual_unet(input_shape=(128,128,3), base_filters=32):
        inputs = Input(shape=input_shape)

        # Encoder
        c1 = conv_block(inputs, base_filters);    p1 = MaxPooling2D((2,2))(c1)
        c2 = conv_block(p1, base_filters*2);      p2 = MaxPooling2D((2,2))(c2)
        c3 = conv_block(p2, base_filters*4);      p3 = MaxPooling2D((2,2))(c3)
        c4 = conv_block(p3, base_filters*8);      p4 = MaxPooling2D((2,2))(c4)

        # Bottleneck
        c5 = conv_block(p4, base_filters*16)

        # Decoder
        u6 = UpSampling2D((2,2))(c5)
        c6 = conv_block(concatenate([u6, c4]), base_filters*8)
        u7 = UpSampling2D((2,2))(c6)
        c7 = conv_block(concatenate([u7, c3]), base_filters*4)
        u8 = UpSampling2D((2,2))(c7)
        c8 = conv_block(concatenate([u8, c2]), base_filters*2)
        u9 = UpSampling2D((2,2))(c8)
        c9 = conv_block(concatenate([u9, c1]), base_filters)

        # (Optional) segmentation head
        seg_output = Conv2D(1, (1,1), activation='sigmoid', name='segmentation')(c9)

        # Return both the full U‑Net and the deepest encoder block c5
        return Model(inputs, seg_output, name='manual_unet'), c5

    ############################################
    # Build a Classification Head on Top of the Encoder
    ############################################
    manual_unet, encoder_output = build_manual_unet(
        input_shape=(128,128,3),    # match your resized image dims
        base_filters=32             # you can lower this if you're still out of RAM
    )
    manual_unet.summary()

    # Now attach the classification head exactly as before:
    x = GlobalAveragePooling2D(name='gap')(encoder_output)
    x = Dense(128, activation='relu', name='cls_fc1')(x)
    x = Dropout(0.5, name='cls_drop')(x)
    class_output = Dense(2, activation='softmax', name='classifier')(x)

    classification_model = Model(
        inputs=manual_unet.input,
        outputs=class_output,
        name='unet_based_classifier'
    )
    classification_model.summary()

    # Compile the classification model.
    optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-4)
    classification_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', dice_coef]
    )


    ############################################
    # Train the Classification Model
    ############################################
    epochs = 1  # Set to 5 here for a quick run; adjust as needed.
    batch_size = 16

    history = classification_model.fit(
        train_seq,
        validation_data=valid_seq,
        epochs=1
    )
    test_loss, test_acc, test_dice = classification_model.evaluate(test_seq, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Dice Coef: {test_dice:.4f}")
    ############################################
    # Evaluate the Model on the Test Set
    ############################################
    print("Test Loss:", test_loss, "Test Acc:", test_acc)

    # 5) predictions + ROC
    y_probs = classification_model.predict(test_seq)
    y_scores = y_probs[:,1]
    y_true = np.concatenate([y for _, y in test_seq], axis=0)

    # Also, get predicted classes for classification report and confusion matrix.


    # Compute confusion matrix and then sensitivity & specificity.
    cm = confusion_matrix(test_labels, predicted_classes)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN)  # True positive rate.
    specificity = TN / (TN + FP)  # True negative rate.
    print("Sensitivity (Recall for positive class): {:.2f}".format(sensitivity))
    print("Specificity (Recall for negative class): {:.2f}".format(specificity))

    ############################################
    # Plot ROC Curve with Sensitivity & Specificity Annotations
    ############################################
    from sklearn.metrics import roc_curve, auc

    # Compute ROC curve: fpr, tpr, thresholds.
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    all_fold_roc.append((fpr, tpr, roc_auc))

    # Specificity is 1 - FPR.
    specificity_vals = 1 - fpr

    # Youden's index = TPR - FPR; find threshold that maximizes it.
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

    # Mark the best threshold point.
    plt.plot(fpr[best_index], tpr[best_index], 'ro', markersize=8)
    plt.text(fpr[best_index] + 0.02, tpr[best_index] - 0.05,
            f'Threshold: {best_threshold:.2f}\nSensitivity: {best_sensitivity:.2f}\nSpecificity: {best_specificity:.2f}',
            fontsize=10, color='red')
    plt.show()

    ############################################
    # Plot Training History
    ############################################
    plt.figure()
    plt.plot(history.history['loss'], 'bo', label='Training Loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.title('Classification Model Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history.history['accuracy'], 'bo', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'b', label='Validation Accuracy')
    plt.title('Classification Model Accuracy')
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
