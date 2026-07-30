# Tabular Data

This example trains Densiformis on mixed-type pandas DataFrames and uses the
trained model to estimate missing-value distributions for validation rows.

## What It Does

The script reads CSV files with pandas, infers each column type, and maps the
columns to Densiformis distribution types:

- `int`, `float` -> `numerical`
- `bool` -> `binary`
- `string`, `object`, `category` -> `categorical`

Numerical columns are normalized with statistics from the training split:

```text
x_normalized = (x - train_mean) / train_std
```

The model is trained in normalized space, then numerical samples are decoded
back to the original units before plotting. Binary and categorical columns are
encoded as scalar binary values or one-hot vectors.

By default, 10% of rows are held out for validation, and 10% of entries in the
training and validation datasets are replaced with `nan` to simulate missing
values during fitting. The final validation plots use clean held-out rows, mask
one target column at a time, and condition on the other observed columns.

## Run
```bash
python main.py train <your_csv_file>.csv
python main.py evaluate <your_csv_file>.csv
```
