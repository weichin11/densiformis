# Text Translation With Conditional Categorical Diffusion

![generated_text_translation_train.gif](https://drive.google.com/thumbnail?id=1cgaK5e5zOvCn_hBkoeH91aHWYRKpeNER&sz=w2560)

This example trains a tiny English/Japanese translation model with the same diffusion API used by the toy conditional examples.

The English and Japanese sentences are tokenized by one shared BPE tokenizer. The model sees two categorical streams, `[english, japanese]`. During translation, one stream is held fixed as the condition with `t_init=0.0`, and the other stream is generated from categorical noise with `t_init=1.0`.

Files:

- `train.tsv`: Parallel English/Japanese train sentence pairs.
- `valid.tsv`: Parallel validation sentence pairs. Its characters are covered by `train.tsv` so the train-fitted tokenizer does not emit `<unk>` for validation examples.
- `train.py`: Trains the shared BPE tokenizer and a two-stream categorical denoiser.
- `translate.py`: Loads a checkpoint, translates examples from `train.tsv` or `valid.tsv`, and writes a GIF visualization.

## Run

Train:

```bash
python examples/text_translation/train.py
```

Translate the first 5 train sentence pairs in both directions:

```bash
python examples/text_translation/translate.py
```

Generate a compact README-sized visualization:

```bash
python examples/text_translation/translate.py --examples 2 --gif-frames 80
```

Compare train and validation examples separately:

```bash
python examples/text_translation/translate.py --split both
```

Translate the first 5 sentence pairs in one direction:

```bash
python examples/text_translation/translate.py --direction en-ja
python examples/text_translation/translate.py --direction ja-en
```

Outputs:

- `generated_text_translation_train.txt` / `generated_text_translation_valid.txt`: Decoded translation samples, named by split.
- `generated_text_translation_train.gif` / `generated_text_translation_valid.gif`: Conditional denoising visualizations, named by split.
- `text_translation_checkpoints/`: Model weights, optimizer state, and `translation_config.json`.
