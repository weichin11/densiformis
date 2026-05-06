# Toy Samples

These examples show how Densiformis handles mixed data types in small synthetic datasets. Each script trains a compact MLP denoiser, samples under several conditioning modes, and writes a grid image of the denoising trajectory.

Files:

- `two_moons.py`: Trains on 2D two-moons coordinates with a binary class label.
- `multi_classification.py`: Trains on 2D synthetic classification data with a 3-class categorical label encoded as one-hot vectors.


## Run

Two moons:

```bash
python examples/toy_samples/two_moons.py
```

Output:

- `moons_grid.png`: Denoising progress for denoising both variables, only positions, or only labels.
![moons_grid.png](https://drive.google.com/thumbnail?id=1ML9ia0cIiOX6zsz1WPG4O3wUMluofA_T&sz=w2560)

Multi-classification:

```bash
python examples/toy_samples/multi_classification.py
```

Output:

- `multi_grid.png`: Denoising progress for denoising both variables, only positions, or only labels.
![multi_grid.png](https://drive.google.com/thumbnail?id=1H-WdPFA6zUs5IHul_rg8OdK7ucaPv1Md&sz=w2560)
