data_dir = "/content/drive/My Drive/New_liver/Training_Batch"

# Initialize a list to store the patient data.
# Each element will be a tuple: (patient_id, ct_volume_with_liver, mask_volume_with_liver)
patient_data = []
i=28

for patient_id in range(i):
    # Construct the file paths
    ct_filename = f"volume-{patient_id}.nii"
    seg_filename = f"segmentation-{patient_id}.nii"

    ct_path = os.path.join(data_dir, ct_filename)
    seg_path = os.path.join(data_dir, seg_filename)

    # Check if both files exist
    if not (os.path.exists(ct_path) and os.path.exists(seg_path)):
        print(f"[WARNING] CT or segmentation file not found for patient {patient_id}. Skipping...")
        continue

    # Load the CT volume and segmentation mask
    print(f"\nLoading patient {patient_id} data...")
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()

    mask_img = nib.load(seg_path)
    mask_data = mask_img.get_fdata()

    # Print shapes of the loaded volumes
    print(f"Patient {patient_id}: CT data shape: {ct_data.shape}")
    print(f"Patient {patient_id}: Segmentation data shape: {mask_data.shape}")

    # Check the segmentation contains exactly the 3 expected labels: 0, 1, and 2
    unique_labels = np.unique(mask_data)
    if set(unique_labels) != {0, 1, 2}:
        print(f"[WARNING] Patient {patient_id}: Expected labels { {0,1,2} } but found {set(unique_labels)}. Skipping patient.")
        continue

    # Identify slices that contain the liver (label 1)
    liver_slices = [s for s in range(mask_data.shape[2]) if np.any(mask_data[:, :, s] == 1)]
    if not liver_slices:
        print(f"Patient {patient_id}: No liver (label 1) found in any slice. Skipping patient.")
        continue

    print(f"Patient {patient_id}: Liver found in slices: {liver_slices}")

    # Visualize the mask slices that contain liver
    num_slices = len(liver_slices)
    cols = 5  # Define number of columns for display; adjust as needed
    rows = math.ceil(num_slices / cols)

    plt.figure(figsize=(cols * 3, rows * 3))
    plt.suptitle(f"Patient {patient_id} - Liver Mask Slices", fontsize=16)
    for i, slice_idx in enumerate(liver_slices):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(mask_data[:, :, slice_idx], cmap='gray')
        plt.title(f"Slice {slice_idx}")
        plt.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Extract the slices with liver from both CT and mask volumes
    ct_volume_with_liver = ct_data[:, :, liver_slices]
    mask_volume_with_liver = mask_data[:, :, liver_slices]

    # Append the data to the list
    patient_data.append((patient_id, ct_volume_with_liver, mask_volume_with_liver))

print("\nData loading and liver slice extraction completed.")
print(f"Total patients processed: {len(patient_data)}")

# Now, all the data for further processing is stored in 'patient_data'

# Assuming patient_data is a list of tuples:
# (patient_id, ct_volume_with_liver, mask_volume_with_liver)
# where each volume has the shape: (height, width, num_slices)

# Step 1: Find the maximum number of slices among all patients.
max_slices = max(ct_vol.shape[2] for _, ct_vol, _ in patient_data)
print("Maximum number of slices across patients:", max_slices)

# Step 2: Initialize an empty list to store the stratified data.
# The new list will store tuples: (patient_id, stratified_ct_volume, stratified_mask_volume)
stratified_data = []

# Step 3: Iterate over each patient's data and stratify to the max number of slices.
for patient_id, ct_vol, mask_vol in patient_data:
    current_slices = ct_vol.shape[2]

    # If current number of slices already equals max_slices, use the volume as is.
    if current_slices == max_slices:
        stratified_data.append((patient_id, ct_vol, mask_vol))
    else:
        # Use np.linspace to generate new indices to sample/duplicate slices.
        new_indices = np.linspace(0, current_slices - 1, num=max_slices)
        new_indices = np.round(new_indices).astype(int)
        # Ensure the indices do not exceed the bounds.
        new_indices[new_indices >= current_slices] = current_slices - 1

        # Create new stratified volumes based on the computed indices.
        new_ct_vol = ct_vol[:, :, new_indices]
        new_mask_vol = mask_vol[:, :, new_indices]

        print(f"Patient {patient_id} stratified from {current_slices} slices to {new_ct_vol.shape[2]} slices.")

        stratified_data.append((patient_id, new_ct_vol, new_mask_vol))

print(f"Stratified CT volume shape: {new_ct_vol.shape}")
print(f"Stratified mask volume shape: {new_mask_vol.shape}")
# The stratified_data list now holds the processed data for further analysis.
print("\nStratification complete.")
print("Total patients in stratified data:", len(stratified_data))

# Define patch size (16x16)
patch_size = 16

# Initialize lists to store the CT patches and their corresponding labels
ct_patches = []
patch_labels = []

# Process each patient in stratified_data
# Each entry in stratified_data is in the format: (patient_id, ct_volume, mask_volume)
for patient_id, ct_vol, mask_vol in stratified_data:
    num_slices = ct_vol.shape[2]
    print(f"Processing patient {patient_id} with {num_slices} slices...")

    # Process each slice individually
    for s in range(num_slices):
        # Extract the current slice from CT and segmentation volumes
        slice_ct = ct_vol[:, :, s]
        slice_mask = mask_vol[:, :, s]

        # Create a binary tumor mask: pixels equal to 2 indicate tumor regions
        tumor_mask = (slice_mask == 2)

        # Loop over non-overlapping 16x16 patches in both dimensions
        for i in range(0, slice_ct.shape[0], patch_size):
            for j in range(0, slice_ct.shape[1], patch_size):
                patch_ct = slice_ct[i:i+patch_size, j:j+patch_size]
                patch_tumor = tumor_mask[i:i+patch_size, j:j+patch_size]

                # Label the patch as tumor (1) if any pixel in the tumor patch is True, else non-tumor (0)
                label = 1 if np.any(patch_tumor) else 0

                ct_patches.append(patch_ct)
                patch_labels.append(label)

# Convert the lists to numpy arrays for further processing.
ct_patches = np.array(ct_patches)
patch_labels = np.array(patch_labels)

print("Extracted CT patches shape:", ct_patches.shape)
print("Extracted patch labels shape:", patch_labels.shape)

# Suppose ct_patches is your grayscale array, e.g., shape (num_patches, 16, 16)
print("Original ct_patches shape:", ct_patches.shape)

# If ct_patches is 3-dimensional (num_patches, H, W), add a new channel dimension
if ct_patches.ndim == 3:
    ct_patches = np.expand_dims(ct_patches, axis=-1)  # Now shape becomes (num_patches, H, W, 1)

# Repeat the grayscale channel 3 times along the last axis to make RGB
ct_patches_rgb = np.repeat(ct_patches, 3, axis=-1)

print("CT patches RGB shape:", ct_patches_rgb.shape)
