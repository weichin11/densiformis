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
### [Two moons](/examples/toy_samples) — 2D points + binary label 
![moons_grid.png](https://drive.google.com/thumbnail?id=1ML9ia0cIiOX6zsz1WPG4O3wUMluofA_T&sz=w2560)

### [Multi-classification](/examples/toy_samples) — 2D points + 3 categorical label
![multi_grid.png](https://drive.google.com/thumbnail?id=1H-WdPFA6zUs5IHul_rg8OdK7ucaPv1Md&sz=w2560)

### [Text Generation](/examples/text_generation)
![generated_text_generation.gif](https://drive.google.com/thumbnail?id=1O5nFE2MVAtAHLwLEbpQOdgFBXvcPpG5O&sz=w2560)


### [Text Translation](/examples/text_translation)
![generated_text_translation_train.gif](https://drive.google.com/thumbnail?id=1cgaK5e5zOvCn_hBkoeH91aHWYRKpeNER&sz=w2560)
