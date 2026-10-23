# Flower Species Recognition using a CNN

A Convolutional Neural Network built **from scratch** (no transfer learning) in TensorFlow/Keras to classify flower photos into 5 species. The project's core idea is to **show overfitting in a baseline CNN, then reduce it** with data augmentation and Dropout, and compare the two models.

> Course project: CA1, Deep Learning (individual project).

---

## Problem Statement

Plant identification is useful for gardening and agriculture apps. This project classifies a flower photo into one of five species: **daisy, dandelion, roses, sunflowers, tulips**.

## Dataset

| Property | Value |
|---|---|
| Source | [TensorFlow Flowers](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) |
| Total images | 3,670 |
| Classes | 5 |
| Train / Validation split | 2,936 / 734 (80/20, fixed seed) |
| Image size | Resized to 128 x 128 |
| Batch size | 32 |

Images per class: daisy 633, dandelion 898, roses 641, sunflowers 699, tulips 799. The classes are mildly imbalanced.

## Approach

1. **Baseline CNN**: no augmentation, no Dropout. Trained for 15 epochs to demonstrate overfitting.
2. **Improved CNN**: the *same architecture* plus data augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`) and Dropout (0.25 after the conv blocks, 0.5 after the dense layer). Trained for 25 epochs.
3. **Comparison**: training curves side by side, confusion matrix, per-class precision/recall/F1, misclassified-image analysis, and a results table.

Both models come from one function, `build_cnn(improved=False/True)`, so the only differences are the techniques being tested.

## Model Architecture

```
Input (128x128x3)
[Improved only: RandomFlip -> RandomRotation -> RandomZoom]
Rescaling (1/255)
Conv2D(32, 3x3, ReLU) -> MaxPooling2D
Conv2D(64, 3x3, ReLU) -> MaxPooling2D
Conv2D(128, 3x3, ReLU) -> MaxPooling2D
[Improved only: Dropout(0.25)]
Flatten
Dense(128, ReLU)
[Improved only: Dropout(0.5)]
Dense(5, softmax)
```

- **Parameters:** 4,288,325 (about 16.4 MB), identical for both models, since augmentation and Dropout add no weights.
- **Loss:** sparse categorical cross-entropy (integer labels)
- **Optimizer:** Adam (default learning rate)
- **Metric:** accuracy

## Results

All numbers below are from a single run of the notebook (TensorFlow 2.21.0, GPU runtime on Google Colab).

| Metric | Baseline CNN | Improved CNN (aug + dropout) |
|---|---|---|
| Epochs | 15 | 25 |
| Final train accuracy | 0.9918 | 0.8045 |
| Final validation accuracy | 0.6689 | 0.7534 |
| Best validation accuracy | 0.6689 | 0.7602 |
| Final train loss | 0.0290 | 0.5012 |
| Final validation loss | 2.2104 | 0.6872 |
| Accuracy gap (train - val) | 0.3229 | 0.0511 |

**Overfitting analysis:** the baseline reaches about 99% training accuracy but only about 67% validation accuracy, and its validation loss climbs from about 1.0 to 2.2 while training loss keeps falling, a clear sign of memorisation. The improved model's train/validation gap is much smaller (about 5 points vs about 32), and its validation accuracy is higher.

### Improved model: per-class results (validation set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| daisy | 0.85 | 0.65 | 0.74 | 129 |
| dandelion | 0.83 | 0.76 | 0.79 | 176 |
| roses | 0.67 | 0.57 | 0.61 | 120 |
| sunflowers | 0.84 | 0.88 | 0.86 | 152 |
| tulips | 0.63 | 0.85 | 0.73 | 157 |
| **Overall accuracy** | | | **0.75** | 734 |

181 of the 734 validation images were misclassified. Roses has the lowest recall and tulips the lowest precision, so those classes are the most confused. See the confusion matrix and misclassified-image grid in the notebook.

Note: results vary slightly between runs because of GPU non-determinism, so rerunning the notebook may give slightly different numbers.

## Repository Contents

| File | Description |
|---|---|
| `Flower_Species_Recognition_Model.ipynb` | Complete notebook: data, both models, plots, evaluation, prediction, saving |
| `flower_cnn.keras` | Saved best model (created when you run the notebook) |
| `README.md` | This file |

## How to Run

The easiest way is Google Colab:

1. Open `Flower_Species_Recognition_Model.ipynb` in [Google Colab](https://colab.research.google.com/) (File > Upload notebook).
2. Select a GPU: **Runtime > Change runtime type > T4 GPU**. CPU also works but is slower.
3. Run all cells top to bottom (**Runtime > Run all**). The dataset downloads automatically (about 220 MB).

To run locally, install the dependencies first:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

The last cell of the "Save and Load" section uses `google.colab.files.download`, which works only in Colab. Remove or comment out those two lines when running locally.

## Notebook Walkthrough

1. Imports and setup (random seeds fixed)
2. Download and inspect the dataset (class counts, bar chart)
3. Sample image grid
4. Load train/validation data (`image_dataset_from_directory`, cache, prefetch)
5. Baseline CNN: build, summary, compile, train
6. Baseline training curves
7. Improved CNN: build, summary, compile, train
8. Improved curves and side-by-side before/after comparison
9. Evaluation: confusion matrix heatmap and classification report
10. Misclassified images and error analysis
11. Prediction on a new image (example: a sunflower photo is predicted as sunflowers with 99.3% confidence)
12. Save and reload the model (`.keras` format)
13. Final results table

## Using the Saved Model

```python
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("flower_cnn.keras")
class_names = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

img = tf.keras.utils.load_img("my_flower.jpg", target_size=(128, 128))
arr = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)  # raw pixels; the model rescales internally
probs = model.predict(arr)[0]
print(class_names[np.argmax(probs)], f"{100 * probs.max():.1f}%")
```

## Limitations

- The CNN is small and trained from scratch on only about 3,000 images, so validation accuracy (about 75%) is well below what pre-trained models typically achieve (not used here, by project design).
- Class imbalance is mild but present (dandelion has the most images).
- Roses and tulips are frequently confused, likely because of similar colours and shapes at low resolution (an observation to verify against the misclassified images, not a proven cause).
- The model was evaluated on one fixed train/validation split from a single run, with no separate test set and no repeated runs, so the numbers carry some uncertainty.
- The baseline and improved models were trained for different numbers of epochs (15 vs 25).

## Future Work

- Add a separate test set and repeat runs with different seeds.
- Tune hyperparameters (learning rate, Dropout rate, filters, image size).
- Add Batch Normalization or an early-stopping callback.
- Try transfer learning with a pre-trained network and compare it against this from-scratch model.
- Collect more or more varied images, especially for the weaker classes (roses, daisy).

## Tech Stack

Python, TensorFlow/Keras, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, Google Colab.

## References

- TensorFlow flowers dataset: https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
- TensorFlow image classification tutorial: https://www.tensorflow.org/tutorials/images/classification
- Keras documentation: https://keras.io/api/

## Author

Scott, Lovely Professional University
