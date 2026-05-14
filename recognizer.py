from PIL import Image
import torch
import numpy as np
import cv2
from torchvision import transforms
import logging
import time

infer_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def _pad_to_square(img, pad_value=0):
    """
    Pads image so that width and height are equal for resizing into 224x224

    Params:
        - img
        - pad_value: value to pad the background
    Returns:
        - out: square image
    """
    h, w = img.shape
    size = max(h, w)
    
    out = np.full((size, size), pad_value, dtype=img.dtype)
    
    dy = (size - h) // 2
    dx = (size - w) // 2

    out[dy:dy + h, dx:dx + w] = img
    return out


def predict_letters(model, device, frame, boxes, classes_map, transform=infer_transform,
                    debug=False, conf_threshold=0.0):
    """
    Run model on crops from frame at boxes

    Params:
        - model: PyTorch model
        - device
        - frame: OpenCV frame
        - boxes: list of (h, w, y, x)
        - classes_map: dataset classes
        - transform: torchvision transform to apply to each crop
        - debug: toggle for debug mode
        - conf_threshold: minimum confidence
    Returns:
        - preds: list of int labels
        - confidences: list of floats of predicted class
        - probs: full probabilities
        - elapsed: pair of inference time and number of crops if debug mode on, otherwise None
    """
    frame = frame.copy()

    if len(boxes) == 0 or frame is None:
        if debug: return [], [], np.zeros((0, len(classes_map)), dtype=float), [0, 0]
        else: return [], [], np.zeros((0, len(classes_map)), dtype=float), None

    crops = []
    for (h, w, y, x) in boxes:
        # Avoiding index errors
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        crop_bgr = frame[y1:y2, x1:x2]

        if debug:
            cv2.imshow("Resized Input", cv2.resize(crop_bgr, (224, 224)))

        # Convert to grayscale for thresholding
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        try:
            crop_blur = cv2.medianBlur(crop_gray, 3)
        except Exception as e:
            logging.debug(f"Median Blur ran into an issue: {e}; Crop Shape: {crop_gray.shape}")
            continue

        # Scale blockSize with crop size so threshold adapts to letter scale
        min_dim = min(crop_blur.shape)
        block_size = max(3, (min_dim // 8) | 1)  # must be odd and >= 3

        # Adaptive thresholding to match crops with training data
        crop_bin = cv2.adaptiveThreshold(
            crop_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block_size,
            C=2
        )

        # Cleaning up the crop so the letter is more easily recognized by the model
        kernel = np.ones((3, 3), np.uint8)
        crop_clean = cv2.morphologyEx(crop_bin, cv2.MORPH_CLOSE, kernel)
        crop_thick = cv2.dilate(crop_clean, kernel, iterations=1)

        # Pad to square to preserve letter aspect ratio before resizing
        crop_square = _pad_to_square(crop_thick, pad_value=0)

        interp = cv2.INTER_AREA if crop_square.shape[0] > 224 else cv2.INTER_LINEAR
        crop_resized = cv2.resize(crop_square, (224, 224), interpolation=interp)

        if debug:
            cv2.imshow("B&W Crop", crop_resized)

        pil = Image.fromarray(crop_resized)

        # Rest of transform required to match training data
        t = transform(pil)

        crops.append(t)

    if len(crops) == 0:
        if debug: return [], [], np.zeros((0, len(classes_map)), dtype=float), [0, 0]
        else: return [], [], np.zeros((0, len(classes_map)), dtype=float), None

    batch = torch.stack(crops).to(device)

    with torch.no_grad():
        if debug:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()

        out = model(batch)

        if debug:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start
            time_data = [elapsed, len(crops)]

        probs = torch.softmax(out, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1).tolist()
        confidences = probs[np.arange(len(preds)), preds].tolist()

    if debug: return preds, confidences, probs, time_data
    else: return preds, confidences, probs, None


def overlay_predictions(frame, boxes, preds, confidences, classes_map, conf_threshold=0.0):
    """
    Overlay for predictions

    Params:
        - frame: OpenCV frame
        - boxes: list of (h, w, y, x)
        - preds: list of int labels
        - confidences: list of float of predicted class
        - classes_map: dataset classes
        - conf_threshold: predictions below this are drawn in a different colour and excluded from word
    Returns:
        - frame
    """
    boxes_preds_confs = list(zip(boxes, preds, confidences))
    boxes_preds_confs.sort(key=lambda bpc: bpc[0][3])

    # For high conf
    word_chars = []
    
    for (h, w, y, x), p, conf in boxes_preds_confs:
        ch = classes_map[p] if isinstance(classes_map, (list, tuple)) else classes_map.get(p, str(p))
        label = f"{ch} {conf:.2f}"
        above_threshold = conf >= conf_threshold
        color = (0, 255, 0) if above_threshold else (0, 140, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if above_threshold:
            word_chars.append(ch)

    word = ''.join(word_chars)
    cv2.putText(frame, word, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
    return frame
