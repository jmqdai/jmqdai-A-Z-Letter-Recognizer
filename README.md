# A-Z Handwritten Letter Recognizer
A short project to recognize A-Z handwritten letters.

## Overview
In this project, I trained and fine-tuned an EfficientNet-B0 classifier on an A-Z handwritten letter CSV dataset provided on Kaggle. I used this model to then predict A-Z handwritten letters seen on a light background through a camera.

### How the Model was Trained
The CSV dataset was split into 80% training and 20% validation with stratified sampling to preserve class balance. Each row in the CSV was converted into an image, then normalized, binarized, resized, and had the input repeated across three channels to match the expected input for the pretrained EfficientNet-B0 model. The images were also augmented with slight rotations and translations for a more robust model.

The model was trained over 20 epochs with batch size of 32. The model was evaluated after each epoch on the validation set, using cross-entropy loss to measure the closeness of the model's predictions. The most accurate model was saved with 99.61% validation accuracy.

### How It Works
Using OpenCV, I specifically look for handwritten letters in a certain defined region of the camera. I segment the captured frames into individual crops for each character. Then, I preprocess the character for recognition with grayscale conversion, followed by median blur, adaptive thresholding, morphological cleaning, and finally resizing to prepare it as a binary black and white image. This processed crop is then passed into the model which infers what handwritten letter it is.

## Requirements

- Python 3.9+
- Webcam

> **Note:** the program supports both GPU and CPU execution. See the installation instructions for more details.

## Installation

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

> **Note:** If you want GPU acceleration, make sure you have a CUDA-compatible version of PyTorch installed. Visit https://pytorch.org/get-started/locally/ to get the right install command for your system.

## Usage

Basic usage:

```bash
python main.py
```

Press **`q`** to quit.

The program also supports the following command-line arguments:
| Argument | Default | Description |
|---|---|---|
| `--ckpt` | `best_model.pth` | Path to the model checkpoint file |
| `--camera_index` | `0` | Camera device index |
| `--rect_h` | `0.45` | Height of the detection region as a fraction of the frame height |
| `--rect_w` | `0.8` | Width of the detection region as a fraction of the frame width |
| `--force_cpu` | off | Force CPU even if a CUDA GPU is available |
| `--conf_threshold` | `0.0` | Minimum confidence (0.0–1.0) for a letter to be included in the assembled word. Uncertain letters still appear on screen but are drawn in orange |
| `--debug` | off | Show extra OpenCV windows: the raw crop, the binary threshold mask, and the preprocessed model input |

## Limitations, Challenges, and Future Improvements

### Fitting the Camera Input to the Model
The provided dataset is in grayscale with a light background. However, it is difficult to replicate the same scenario when reading off a piece of paper or a sign in a camera frame due to shadow, lighting differences, natural colour of various materials, and more. Thus, I decided to binarize the camera input and the model training data, providing a much easier way to match the camera input and training data so the model can infer the letters properly.

### Improving Accuracy
Although binarizing the data helps a lot with the model's accuracy, it is not perfect. In particular, there is often a lot of noise, causing either black specs to appear in the inputs, or the letters to have some white spacing in them. These end up confusing the model, causing it to detect incorrect letters. There are a few ways to improve this, such as having the model train on noisier images.

### Expanding the Model
Currently the model only recognizes capital letters. By adding more training data, we can expand this to lowercase letters and numbers as well. In addition, as we used a pretrained EfficientNet-B0 model, it is easier for us to modify the training script to process coloured images. This could significantly alleviate noise issues resulting from binarization of camera inputs.

## Image Sample
<img width="952" height="756" alt="demo-hello" src="https://github.com/user-attachments/assets/6bb9fe00-a80a-4ca0-b4f7-2fa7803f20e3" />


## Resources
### Dataset
https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format
