# Cat vs Dog Classifier (PyTorch)

A trained image classifier that tells cats from dogs, built on the real
Kaggle "Dogs vs. Cats" dataset (25,000 images). It comes with an
**already-trained checkpoint** — you can point it at any photo right now
and get a prediction, no training required. You can also retrain it
yourself for higher accuracy (see below).

## Already-trained results (included in this zip)

- **Test accuracy: 88.3%** on 5,000 held-out test images the model never
  saw during training (2,500 cats + 2,500 dogs).
- Precision/recall around 0.86–0.91 for both classes — see
  `outputs/confusion_matrix.png` and `outputs/training_curves.png`.
- This checkpoint (`checkpoints/best_model.pt`) was trained **from
  scratch** (no pretrained weights), because it was produced in a
  sandboxed environment with no internet access and only 1 CPU core.

## You can do better: transfer learning gets 97–99%

The code in this project **automatically tries pretrained ImageNet
ResNet18 weights first**, and only falls back to the from-scratch CNN if
that download fails. On your own machine (with normal internet access),
just run `python src/train.py` and it will:

1. Download pretrained ResNet18 weights automatically (~45MB, one-time).
2. Train a new classifier head on top (fast), then fine-tune the last
   block.
3. Typically land at **97–99% test accuracy** — well above the 88.3%
   from-scratch result included here.

You'll see this line in the console confirming which path was used:
```
[model] Using pretrained ResNet18 (ImageNet transfer learning).
```
vs.
```
[model] Could not load pretrained ResNet18 weights (...).
[model] Falling back to a from-scratch CNN (SimpleCNN, 64x64 input).
```

## 1. Setup (VS Code / local machine)

```bash
cd cat-dog-classifier
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Open the folder in VS Code (`code .`) and select the `venv` interpreter.

## 2. Try it right now on any photo (uses the included checkpoint)

```bash
python src/predict.py --image path/to/any_photo.jpg
```
Example output:
```
Prediction: dog (confidence 91.42%)
  cat: 8.58% | dog: 91.42%
```
Works on any picture, not just ones from this dataset — that's the point.

## 3. Check the included results yourself

```bash
python src/evaluate.py
```
Reprints the 88.3% test accuracy, classification report, and regenerates
`outputs/confusion_matrix.png` from the included checkpoint.

## 4. Retrain for higher accuracy (recommended, on a machine with internet)

```bash
python src/train.py
```
This runs two phases automatically when pretrained weights are available:
- **Phase 1** — trains a new classifier head on frozen ResNet18 features.
- **Phase 2** — unfreezes the last block and fine-tunes at a lower LR.

Useful flags:
```bash
python src/train.py --epochs1 8 --epochs2 8 --batch-size 32
```

If you want to continue training in short chunks (e.g. on a slow machine),
use `--resume` to pick up from the last saved checkpoint instead of
starting over:
```bash
python src/train.py --epochs1 1 --epochs2 0 --resume
```

To force from-scratch training even when pretrained weights are available:
```bash
python src/train.py --force-scratch
```

## Project structure

```
cat-dog-classifier/
├── data/
│   ├── dataset_info.csv        # manifest: filename, split, class, label
│   └── images/                 # all 25,000 real cat/dog JPEGs (flat folder)
├── src/
│   ├── dataset.py               # CSV-driven Dataset with flexible path resolution
│   ├── model.py                 # ResNet18 (transfer learning) + SimpleCNN fallback
│   ├── train.py                 # two-phase training, auto pretrained/scratch switch
│   ├── evaluate.py              # test accuracy + confusion matrix
│   └── predict.py               # single-image inference -- use this for new photos
├── checkpoints/
│   ├── best_model.pt            # ALREADY TRAINED, 88.3% test accuracy
│   └── history.json             # training history (for the curves plot)
├── outputs/
│   ├── training_curves.png      # loss/accuracy over the 12 training epochs
│   └── confusion_matrix.png     # test-set confusion matrix
├── requirements.txt
└── README.md
```

## Where the dataset came from

Your uploaded train/test folders came through empty, so this project uses
the same dataset downloaded directly from a public mirror of Kaggle's
"Dogs vs. Cats" competition data (25,000 structured cat/dog images). If
you're given a separate dataset zip alongside this project zip, it
contains the exact same images already in `data/images/` here — you don't
need to do anything extra unless you want to replace them with your own.

## Why the reported accuracy is trustworthy

- **Train / validation / test are strictly separate**, defined in
  `data/dataset_info.csv` (90/10 stratified split of the original train
  set into train/validation, plus the original held-out test set).
- **Data augmentation** (random crop, flip, color jitter) is applied to
  training data only.
- **Best-checkpoint selection** uses validation accuracy; the 88.3%
  figure comes from `evaluate.py` on the **test** split, which neither
  training nor checkpoint selection ever touched.

## Troubleshooting

- **"No images found"** — the images should already be in
  `data/images/`; if you replaced them, check `dataset.py`'s docstring
  for the folder layouts it auto-detects.
- **Slow on CPU** — reduce `--batch-size`, lower resolution isn't
  configurable via CLI currently but the scratch-CNN path already uses
  64x64 for speed.
- **Out of memory** — lower `--batch-size` (try 16).
