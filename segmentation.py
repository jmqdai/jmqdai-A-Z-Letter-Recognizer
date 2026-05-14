import cv2
import numpy as np

def get_screen_rect(frame_shape, height = 0.4, width = 0.8):
    """
    Returns centered rect relative to frame shape of height/width of certain percentage of frame shape
    Params:
        - frame_shape
        - height: percent of the frame_shape to be height of rect
        - width: percent of the frame_shape to be width of rect
    Returns:
        - (rect height, rect width, rect posy, rect posx)
    """
    camera_height = frame_shape[0]
    camera_width = frame_shape[1]

    h = int(camera_height * height)
    w = int(camera_width * width)

    y = (camera_height - h) // 2
    x = (camera_width - w) // 2

    return (h, w, y, x)

def draw_rect(frame, rect, colour = (0, 255, 0), thickness = 2, alpha = 0.15):
    """
    Draw highlighted rectangle to position text in

    Params:
        - frame: image to draw rect in
        - rect: the (h, w, y, x) rect
        - colour: the BGR rect colour
        - alpha: transparency of overlay
    """
    h, w, y, x = rect
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, thickness)

def segment_letters_in_box(img, height = 0.8, width = 0.4):
    """
    Get fixed box in img and segment the letters
    
    Params:
        - img
        - height: percent of the img to be height of rect
        - width: percent of the img to be width of rect
    Returns:
        - crop: cropped image
        - boxes_frame: list of (h, w, y, x) boxes in original frame coord sys
        - th_crop: binary mask from crop for debugging
    """
    camera_height = img.shape[0]
    camera_width = img.shape[1]

    h, w, y, x = get_screen_rect((camera_height, camera_width), height = height, width = width)

    crop = img[y:y+h, x:x+w].copy()
    boxes_crop, th_crop = segment_letters(crop)

    boxes_frame = [(bh, bw, by + y, bx + x) for (bh, bw, by, bx) in boxes_crop]

    return crop, boxes_frame, th_crop

def segment_letters(img):
    """
    Function that segments the letters

    Params:
        - img
    Returns:
        - boxes: list of (h, w, y, x) boxes
        - th: binary mask for debugging
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations = 1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    height, width = th.shape
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        
        # Probably noise
        if area < 100:
            continue

        # Too short, probably not letters
        if h < 0.2 * height:
            continue

        # Too large, such as a shadow
        if h > 0.95 * height and w > 0.95 * width:
            continue

        boxes.append((h,w,y,x))

    boxes = sorted(boxes, key=lambda b: b[3])
    return boxes, th
