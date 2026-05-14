import cv2
import torch
import argparse
import logging

from model import LetterClassifier, load_checkpoint
from segmentation import get_screen_rect, draw_rect, segment_letters_in_box
from recognizer import predict_letters, overlay_predictions

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    # Used to log average inference time in debug mode
    total_inference_time = 0
    total_crops = 0
    
    # Controls update frequency
    update_every = 1 if args.debug else 10
    frame_cnt = 0

    # Map from output to character
    num_classes = 26
    classes_map = [chr(ord('A') + i) for i in range(num_classes)]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logging.info(f"Using device: {device}")

    model = LetterClassifier(num_classes=num_classes)
    model = load_checkpoint(model, args.ckpt, device)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        return

    print("Press 'q' to quit")

    # Last predictions stored to be displayed on frames where there is no inference
    last_boxes, last_preds, last_confs = [], [], []
    crop, th = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame from camera.")
            break

        rect = get_screen_rect(frame.shape, height=args.rect_h, width=args.rect_w)
        draw_rect(frame, rect, alpha=0.12)

        if frame_cnt % update_every == 0:
            crop, boxes, th = segment_letters_in_box(frame, height=args.rect_h, width=args.rect_w)

            preds, confs, probs, time_data = predict_letters(
                model, device, frame, boxes, classes_map,
                debug=args.debug, conf_threshold=args.conf_threshold
            )
            
            if args.debug:
                inference_time = time_data[0]
                num_crops = time_data[1]
                # Only count inferences that detect letters
                if (num_crops > 0):
                    logging.debug(f"Inference time: {inference_time:.3f}s, Crops: {num_crops}, Average inference per crop: {(inference_time / num_crops):.3f}s")
                    total_inference_time += inference_time
                    total_crops += num_crops

            last_boxes, last_preds, last_confs = boxes, preds, confs

        frame_cnt = (frame_cnt + 1) % update_every

        overlay_predictions(frame, last_boxes, last_preds, last_confs, classes_map,
                            conf_threshold=args.conf_threshold)
        cv2.imshow("Live", frame)

        if args.debug and crop is not None:
            cv2.imshow("Crop", crop)
            cv2.imshow("Thresh", th)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            if total_crops > 0: logging.debug(f"Average inference time per crop: {(total_inference_time / total_crops):.3f}s")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="best_model.pth", help="path to .pth checkpoint")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--rect_h", type=float, default=0.45)
    p.add_argument("--rect_w", type=float, default=0.8)
    p.add_argument("--force_cpu", action="store_true", help="force CPU even if CUDA is available")
    p.add_argument("--conf_threshold", type=float, default=0.0,
                   help="minimum confidence to include a prediction in the word (0.0 = show all)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    main(args)
