# CycleGAN-Based Lunar Surface Detail Extraction from Permanently Shadowed Regions

![Lunar PSR Example](https://www.nasa.gov/sites/default/files/styles/full_width/public/thumbnails/image/shadowcam_shackleton_crater.jpg)  
*Shackleton Crater's permanently shadowed floor, imaged by NASA's ShadowCam. [Credit: NASA/KARI/ASU]*

---

## Overview

This repository presents a **CycleGAN-based deep learning framework** for extracting surface details from the **Permanently Shadowed Regions (PSRs)** of the lunar surface. These areas, located near the Moon's poles, are extremely cold and dark, making them scientifically valuable but hard to image with conventional techniques.

Our approach leverages **unpaired image-to-image translation** to learn mappings between illuminated lunar terrains and their shadowed, low-visibility counterparts. By combining advanced neural architectures and a suite of custom loss functions, our model predicts plausible geological features in PSRs, balancing generative creativity with scientific plausibility and artifact suppression.

---

## Problem Context

- **Permanently Shadowed Regions (PSRs):**  
  Deep craters near the lunar poles that have not seen sunlight in over two billion years, remaining some of the coldest and least explored places in the solar system.  
  ![Lunar South Pole PSRs](https://moon.nasa.gov/system/resources/detail_files/97_psr_annotated.jpg)  
  *Annotated map of PSRs at the lunar south pole. [Credit: NASA]*

- **Scientific importance:**  
  PSRs may contain preserved water ice and ancient volatiles, critical for future lunar exploration.

- **Imaging challenge:**  
  Traditional enhancement methods amplify noise or miss subtle geological features. New deep learning approaches are needed to reveal hidden details.

---

## CycleGAN Architecture

### Core Principles

- **Two Generators:**  
  - `G_AB`: Illuminated → Shadowed  
  - `G_BA`: Shadowed → Illuminated

- **Two Discriminators:**  
  - `D_A`: Real vs. Fake in Illuminated domain  
  - `D_B`: Real vs. Fake in Shadowed domain

- **Cycle Consistency:**  
  Ensures that translating an image from one domain to another and back should recover the original.

![CycleGAN Schematic](https://raw.githubusercontent.com/junyanz/CycleGAN/master/docs/arch.png)  
*CycleGAN architecture: two generators and two discriminators with cycle consistency loss. [Source: CycleGAN paper]*

---

## How CycleGAN Works

CycleGAN enables **unpaired image-to-image translation** by learning two mappings (A→B and B→A) and enforcing *cycle consistency*—if you translate an image to the other domain and back, you should get the original image.  
- **Adversarial Loss:** Each generator tries to fool its corresponding discriminator.
- **Cycle Consistency Loss:** Ensures reconstructed images resemble the originals.
- **Identity Loss:** Prevents unnecessary changes when the input is already in the target domain.

> “The cycle consistency loss ensures that an image from one domain, when translated to the other domain and then back, is similar to the original image. Using this loss makes the model preserve the underlying structure and content of the image and learn useful semantic representation and not output random images.”  
> — [viso.ai, 2025][4]

---

## Custom Loss Functions

Our implementation uses a sophisticated loss strategy to balance realism and scientific validity:

- **Adversarial Loss:** Standard GAN loss for realism.
- **Cycle Consistency Loss:**  
  - **L1 Loss:** Pixel-wise similarity.  
  - **Perceptual Loss:** Semantic similarity via VGG features.  
  - **SSIM Loss:** Structural similarity for textures and edges.
- **Identity Loss:** Same components as above, to preserve input characteristics.

**Total Loss Formula:**
