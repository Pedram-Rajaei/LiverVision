# LiverVision: Liver Tumor Detection and Segmentation in CT with a Patch-Based VAE

![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Dataset](https://img.shields.io/badge/Data-LiTS-blue)
![Type](https://img.shields.io/badge/Course-project-lightgrey)

This project tackles the challenge of accurate, efficient liver tumor segmentation in CT scans by combining variational inference with convolutional decoding. Our pipeline learns a compact latent representation of small image patches, then reconstructs high-resolution tumor masks and spatial coordinates—automating what is traditionally a time-consuming, error-prone task for radiologists.

## Introduction

Early detection of liver cancer greatly improves patient outcomes, yet manual annotation of CT scans is laborious and subjective. By framing segmentation as a variational inverse problem, our approach uses a patch-based Variational Autoencoder (VAE) encoder to capture meaningful tissue features and uncertainty, and a transpose-convolution decoder to reconstruct precise tumor boundaries and output their pixel-level coordinates

## Related Work

Prior methods like U-Net and cascaded FCN+3D-CRF achieve strong segmentation but require large annotations, multi-stage processing, and offer no uncertainty quantification. Probabilistic U-Net introduces latent modeling but still relies on separate refinement steps. Our end-to-end VAE–transpose-convolution pipeline unifies uncertainty regularization with direct mask and coordinate output, reducing complexity and data requirements.

## Data Preparation

We used CT volumes from the public [LiTS challenge dataset](https://arxiv.org/abs/1901.04056) (Liver Tumor Segmentation Benchmark). As a course project with limited compute, the experiments use a subset of **6 patients**, evaluated with patient-level leave-one-out cross-validation (in each fold: 4 patients for training, 1 for validation, 1 held out for testing).

From the liver region of each slice, we extracted 16×16-pixel patches and labeled a patch as "tumor" if ≥ 30% of its pixels belong to the tumor mask, otherwise "non-tumor". To mitigate patient-level imbalance, we stratified each patient to the same number of slices by duplicating under-represented volumes.

## Methods

1. **VAE Encoder**  
   Two convolutional layers downsample each patch to an 8×8 feature map, followed by dense layers that output the latent mean μ and log-variance log σ².  
2. **Reparameterization**  
   Latent vectors \(z = μ + σ⋅ϵ\) allow stochastic sampling while preserving gradient flow.  
3. **Transpose-Convolution Decoder**  
   Two Conv2DTranspose layers upsample z back to 16×16, producing both a binary mask and explicit (x,y) coordinates for tumor regions.  
4. **Patch Classifier**  
   A separate branch uses z, a Conv2DTranspose upsampling, and two Dense layers (with dropout and L2 regularization) to predict tumor presence via softmax.  
5. **Losses & Training**  
   - VAE: MSE reconstruction + KL divergence to unit Gaussian  
   - Classifier: Cross-entropy on patch labels  
   We pretrained the VAE for 100 epochs, then trained the classifier (50–100 epochs) under a cyclical learning rate schedule with AdamW and early stopping.

## Results

Using six-fold leave-one-out cross-validation on six patients, the model's patch-level tumor detection reached AUC ≥ 0.90 on four of six held-out patients (overall competitive with U-Net and ResNet-50), while using ≈ ½ the memory and ⅔ the training time of U-Net. Qualitative examples show accurate delineation of tumor regions and highlight failure modes (e.g., blood-vessel false positives).

## Discussion & Future Work

- **Advantages**: Lightweight, end-to-end uncertainty modeling, direct coordinate output.  
- **Limitations**: Patch-based context loss, 2D slice independence, small cohort validation.  
- **Next Steps**:  
  - Extend to 3D VAE for volumetric consistency  
  - Multi-task learning (joint liver/vessel/tumor segmentation)  
  - Incorporate attention mechanisms and pretrained backbones for hybrid latent modeling  

By automating and regularizing liver tumor delineation, this VAE-based pipeline promises faster, more consistent diagnoses and lays groundwork for real-time, edge-device inference in resource-constrained clinical settings.

## Repository Structure

```
LiverVision/
├── Data_Preprocessing.py   # Load LiTS volumes (NIfTI), keep liver slices, build patient-level data
├── LiverVision.py          # VAE encoder, transpose-conv decoder and patch classifier; training and evaluation
├── LOOCV_LiverVision.py    # Patient-level leave-one-out cross-validation of the full pipeline
├── UNET_model.py           # U-Net baseline
└── ResNet_model.py         # ResNet-50 baseline
```

## Code Status

The scripts are exported from the Google Colab notebooks used for the project, so they read data from Google Drive paths (`/content/drive/...`). To run them, download LiTS and point `data_dir` at your local copy.

Dependencies: `tensorflow`, `numpy`, `pandas`, `nibabel`, `h5py`, `opencv-python`, `scikit-learn`, `matplotlib`, `seaborn`, and `segmentation-models` (U-Net baseline only).

## Reference

Bilic, P. et al. "The Liver Tumor Segmentation Benchmark (LiTS)." *Medical Image Analysis* 84 (2023): 102680. [arXiv:1901.04056](https://arxiv.org/abs/1901.04056)
