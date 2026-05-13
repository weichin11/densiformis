# Image Generation

![celeb_a_mask_hq_grid.png](https://drive.google.com/thumbnail?id=1f19zch_dfRViV703z2Z1tyPD1-ZybyzH&sz=w2560)

These examples show how Densiformis handles image-like tensors together with structured visual conditions. The CelebA Mask-HQ example models a face image as a numerical distribution and its semantic segmentation mask as a binary distribution, then samples under several conditioning modes.

Files:

- `celeb_a_mask_hq.py`: Loads CelebA Mask-HQ images and masks from a zip archive, defines a U-Net denoiser, and writes a grid of generated image/mask samples.

## Data

Download [CelebA Mask-HQ](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view) and place the archive here:

```bash
data/CelebAMask-HQ.zip
```

The script expects the zip archive to contain:

- `CelebAMask-HQ/CelebA-HQ-img/`
- `CelebAMask-HQ/CelebAMask-HQ-mask-anno/`

## Run

```bash
python examples/image_generation/celeb_a_mask_hq.py
```

The script loads checkpoints from the repository root:

- `model.pt`
- `optimizer.pt`

To train instead of only generating from an existing checkpoint, uncomment the `densiformis.diffuser.fit(...)` block in `celeb_a_mask_hq.py`.

Output:

- `celeb_a_mask_hq_grid.png`: Denoising progress for generating both image and mask, generating images with masks fixed, and generating masks with images fixed.
