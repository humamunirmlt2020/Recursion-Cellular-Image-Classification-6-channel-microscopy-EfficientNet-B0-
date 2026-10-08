# Recursion Cellular Image Classification (6-channel microscopy, EfficientNet-B0)

Deep learning pipeline that predicts which siRNA (genetic perturbation) was applied to a well of HepG2 cells, using 6-channel fluorescence microscopy images. Built in PyTorch on the Kaggle [Recursion Cellular Image Classification](https://www.kaggle.com/competitions/recursion-cellular-image-classification) dataset.

## Competition background

Bringing a new treatment to market takes more than 10 years on average and costs hundreds of millions of dollars, which is one reason drug prices keep rising. Recursion Pharmaceuticals, which has generated a large in-house dataset of biological images, ran this competition to see whether machine learning can help model how genetic perturbations affect human cells.

The core challenge is separating real biological signal from experimental noise: technical execution and environmental variation between experiments (batch effects) change how the images look. Entries classify images of cells under one of 1,108 genetic perturbations (siRNAs).

The competition was part of the NeurIPS 2019 competition track and closed on September 26, 2019. More background on the dataset (RxRx1) is on [rxrx.ai](https://www.rxrx.ai), and the host's starter code is in [recursionpharma/rxrx1-utils](https://github.com/recursionpharma/rxrx1-utils).

## Problem

Each well is imaged at two sites (`s1`, `s2`) in 6 fluorescence channels (`w1`-`w6`). The task is multi-class classification into **1,108 siRNA classes**. Train set: 36,517 wells. Test set: 19,899 wells.

### Dataset files

The Kaggle dataset is about 48.94 GB:

| File / folder | Contents |
|---|---|
| `train/` | Training images, one PNG per channel per site per well |
| `test/` | Test images, same layout |
| `train.csv` | Training wells with experiment, plate, well and `sirna` label |
| `test.csv` | Test wells (no label) |
| `train_controls.csv` | Control wells for the training experiments |
| `test_controls.csv` | Control wells for the test experiments |
| `pixel_stats.csv` | Per-image, per-channel pixel statistics (mean, std, etc.) |
| `sample_submission.csv` | Submission template |
| `recursion_dataset_license.pdf` | Dataset licence |

### Evaluation

Submissions are scored on **multiclass accuracy**: the fraction of test wells whose `sirna` is predicted correctly.

### Submission format

One row per `id_code` in the test set, with a header:

```
id_code,sirna
HEPG2-08_1_B03,911
HEPG2-08_1_B04,911
```

## Method

- **Input:** all 6 channels stacked as one 6-channel tensor, resized from 512x512 to 384x384, scaled to [0, 1].
- **Both sites used:** each well becomes two rows (`s1`, `s2`). Predicted probabilities are averaged per well at validation and test time.
- **Model:** ImageNet-pretrained EfficientNet-B0. The first convolution is replaced with a 6-channel version, initialised by copying the pretrained RGB weights to channels 1-3 and 4-6 (scaled by 0.5).
- **Validation:** 5-fold `GroupKFold` grouped by **experiment**, to reduce leakage between train and validation. Only fold 0 was trained and reported.
- **Training:** AdamW (lr 3e-4, weight decay 1e-4), cosine annealing, mixed precision, batch size 16, 10 epochs, seed 42. Augmentation: random horizontal/vertical flips and 90-degree rotations.
- **Inference:** 4-way test-time augmentation (identity, horizontal flip, vertical flip, 90-degree rotation) averaged with the two sites.
- **Metrics:** accuracy, top-5 accuracy, macro / weighted / micro F1.

## Results (validation, fold 0, epoch 10)

| Metric | Value |
|---|---|
| Accuracy | 0.546 |
| Top-5 accuracy | 0.731 |
| Macro F1 | 0.538 |
| Validation loss | 2.684 |

Random guessing over 1,108 classes would give about 0.09% accuracy. Training accuracy at epoch 10 was 0.719, and validation loss was still decreasing, so the model had not fully converged.

The full training log and curves are in the notebook outputs.

## Limitations

- Only one fold and one run; no confidence intervals.
- 10 epochs only; validation performance was still improving.
- Preprocessing is minimal (resize and divide by 255). No per-experiment or per-plate normalisation, no use of control wells, and no illumination correction. These are known to matter for this dataset (batch effects between experiments).
- Missing image files are replaced with black images; the number of affected files was not counted.
- This is a learning baseline, not a competitive leaderboard solution.

## Possible next steps

- Per-experiment / per-channel normalisation and use of control wells.
- Train all 5 folds and ensemble.
- Longer training and a stronger backbone.

## How to run

1. Open the notebook on Kaggle and add the competition dataset, or download it from the competition page.
2. If running locally, change `data_path` in the `CFG` class to your dataset location.
3. A GPU is needed for reasonable runtime (trained on a Tesla T4).
4. Run the notebook top to bottom. It saves `best_model.pth`, `training_history.csv` and `submission.csv`.

```
pip install -r requirements.txt
```

The dataset is not included in this repository.

## Acknowledgements

- **Data:** the RxRx1 dataset is provided by [Recursion Pharmaceuticals](https://www.recursion.com) through the [Kaggle competition](https://www.kaggle.com/competitions/recursion-cellular-image-classification), under the licence in `recursion_dataset_license.pdf`. It is not redistributed here.
- **Pretrained weights:** EfficientNet-B0 ImageNet weights from [torchvision](https://pytorch.org/vision/stable/models/efficientnet.html) (originally trained by Ross Wightman).
- **Libraries:** PyTorch, torchvision, scikit-learn, pandas, NumPy, Pillow, Matplotlib and tqdm.
- **Compute:** trained on a Kaggle GPU (Tesla T4).

## Citation

Ben Mabey, Berton, Gad Benram, Jane Hung, Jason Yosinski, Julia Elliott, Recursion James, Rosanne Liu, and Sohier Dane. *Recursion Cellular Image Classification*. https://www.kaggle.com/competitions/recursion-cellular-image-classification, 2019. Kaggle.

## Author

Huma
