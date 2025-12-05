# Densiformis

A lightweight toolkit for training multimodal and conditional generative diffusion models across mixed data types (numerical, binary, categorical).

## Installation

### 1. Create and activate an environment

```bash
conda create -n densiformis python=3.12
conda deactivate
conda activate densiformis
```

### 2. Clone and install

```bash
git clone https://github.com/weichin11/densiformis.git
cd densiformis
pip install -e .
```

### 3. Install example extras

```bash
pip install -e .[examples]
```

## Examples
### Two moons — 2D points + binary label

```bash
python examples/two_moons.py
```

Output: `moons_grid.png` shows denoising progress when denoising both variables, only positions, or only labels.
![moons_grid.png](https://drive.google.com/thumbnail?id=1ML9ia0cIiOX6zsz1WPG4O3wUMluofA_T&sz=w2560)

### Multi-classification — 2D points + 3 categorical label (one-hot encoded)
```bash
python examples/multi_classification.py
```

Output: `multi_grid.png` visualizes denoising under the three conditioning modes.
![multi_grid.png](https://drive.google.com/thumbnail?id=1H-WdPFA6zUs5IHul_rg8OdK7ucaPv1Md&sz=w2560)
