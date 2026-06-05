# Image Generation

These examples show how Densiformis handles image-like tensors together with structured visual conditions. Start with `mnist.py` for a small image/class-label example, then move to `celeb_a_mask_hq.py` for paired face images and semantic segmentation masks.

## MNIST

![mnist](https://drive.google.com/thumbnail?id=1WMWjLDAu3X39tue-I3v-d6yDS7bxpqJo&sz=w2560)

`mnist.py` models each handwritten digit image as a numerical distribution and its class label as a categorical distribution. The generated video shows three conditioning modes:

- Generate both images and labels from noise.
- Generate images while keeping labels fixed.
- Generate labels while keeping images fixed.

Data is downloaded automatically by torchvision into `data/`.

Run:

```bash
python examples/image_generation/mnist.py
```

Output:
- `mnist_grid.mp4`: Denoising progress for image and label generation.

## CelebA Mask-HQ

![celeb_a_mask_hq_grid](https://drive.google.com/thumbnail?id=1CcalriiXwWQgynhzOzlcKBCEbgsvMQ95&sz=w2560)

`celeb_a_mask_hq.py` models a face image as a numerical distribution and its semantic segmentation mask as a binary distribution. The generated video shows three conditioning modes:

- Generate both images and masks from noise.
- Generate images while keeping masks fixed.
- Generate masks while keeping images fixed.

Download [CelebA Mask-HQ](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view) and place the archive here:

```bash
data/CelebAMask-HQ.zip
```

The script expects the zip archive to contain:

- `CelebAMask-HQ/CelebA-HQ-img/`
- `CelebAMask-HQ/CelebAMask-HQ-mask-anno/`

Run:

```bash
python examples/image_generation/celeb_a_mask_hq.py
```

Output:

- `celeb_a_mask_hq_grid.mp4`: Denoising progress for image and mask generation.
