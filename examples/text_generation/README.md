# Text Translation With Conditional Categorical Diffusion

![generated_text_translation_train.gif](https://drive.google.com/thumbnail?id=1cgaK5e5zOvCn_hBkoeH91aHWYRKpeNER&sz=w2560)

This example trains a tiny English/Japanese translation model with the same diffusion API used by the toy conditional examples.

The English and Japanese sentences are tokenized by one shared BPE tokenizer. The model sees two categorical streams, `[english, japanese]`. During translation, one stream is held fixed as the condition with `t_init=0.0`, and the other stream is generated from categorical noise with `t_init=1.0`.
The train and validation splits are derived from `translate.csv` with `--valid-fraction`.

Files:

- `translate.csv`: Parallel English/Japanese sentence pairs.
- `translate.py`: Trains the shared BPE tokenizer and two-stream categorical denoiser, or generates translations from a checkpoint.

## Run

Train:

```bash
python examples/text_generation/translate.py --phase train
```

Generate translations for the first 3 train sentence pairs in both directions:

```bash
python examples/text_generation/translate.py --phase generate
```

Generate a compact README-sized visualization:

```bash
python examples/text_generation/translate.py --phase generate --examples 2 --frames 80
```

Generate an MP4 visualization:

```bash
python examples/text_generation/translate.py --phase generate --output-path generated_text_translation.mp4 --frames 80 --fps 20
```

Compare train and validation examples separately:

```bash
python examples/text_generation/translate.py --phase generate --split both
```

Generate translations for the first 3 sentence pairs in one direction:

```bash
python examples/text_generation/translate.py --phase generate --direction en-ja
python examples/text_generation/translate.py --phase generate --direction ja-en
```

Outputs:

- `generated_text_translation_train.txt` / `generated_text_translation_valid.txt`: Decoded translation samples, named by split.
- `generated_text_translation_train.gif` / `generated_text_translation_valid.gif`: Conditional denoising visualizations, named by split. Use `--output-path generated_text_translation.mp4` for MP4 output.
- `text_translation_checkpoints/`: Model weights, optimizer state, and `translation_config.json`.
