# LiverVision: Automated Liver Tumor Segmentation via Variational Autoencoder

This project tackles the challenge of accurate, efficient liver tumor segmentation in CT scans by combining variational inference with convolutional decoding. Our pipeline learns a compact latent representation of small image patches, then reconstructs high-resolution tumor masks and spatial coordinates—automating what is traditionally a time-consuming, error-prone task for radiologists.

## Introduction

Early detection of liver cancer greatly improves patient outcomes, yet manual annotation of CT scans is laborious and subjective. By framing segmentation as a variational inverse problem, our approach uses a patch-based Variational Autoencoder (VAE) encoder to capture meaningful tissue features and uncertainty, and a transpose-convolution decoder to reconstruct precise tumor boundaries and output their pixel-level coordinates

## Related Work

Prior methods like U-Net and cascaded FCN+3D-CRF achieve strong segmentation but require large annotations, multi-stage processing, and offer no uncertainty quantification. Probabilistic U-Net introduces latent modeling but still relies on separate refinement steps. Our end-to-end VAE–transpose-convolution pipeline unifies uncertainty regularization with direct mask and coordinate output, reducing complexity and data requirements.

## Data Preparation

We leveraged the public LiTS challenge dataset (130 training, 70 test scans). To mitigate patient-level imbalance, we stratified each patient to the same number of slices by duplicating under-represented volumes. From the liver region of each slice, we extracted 16×16-pixel patches and labeled them as “tumor” if ≥ 30 % of pixels belong to tumor masks, otherwise “non-tumor”.

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

Using six-fold leave-one-out cross-validation on six patients, our model achieved AUC ≥ 0.90 on four of six folds (overall competitive with U-Net and ResNet-50), while using ≈ ½ the memory and ⅔ the training time of U-Net. Qualitative examples show accurate delineation of tumor regions and highlight failure modes (e.g., blood-vessel false positives).

## Discussion & Future Work

- **Advantages**: Lightweight, end-to-end uncertainty modeling, direct coordinate output.  
- **Limitations**: Patch-based context loss, 2D slice independence, small cohort validation.  
- **Next Steps**:  
  - Extend to 3D VAE for volumetric consistency  
  - Multi-task learning (joint liver/vessel/tumor segmentation)  
  - Incorporate attention mechanisms and pretrained backbones for hybrid latent modeling  

By automating and regularizing liver tumor delineation, this VAE-based pipeline promises faster, more consistent diagnoses and lays groundwork for real-time, edge-device inference in resource-constrained clinical settings.  
