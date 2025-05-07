import cv2
import os
import argparse
import numpy as np
import json
import glob
from tqdm import tqdm
import torch
from transformers import AutoModel
import tempfile
import shutil
import zipfile
try:
    import rarfile
except ImportError:
    rarfile = None

def split_panels(image_path, output_dir, min_panel_area_percent=2.0, canny_threshold1=10, canny_threshold2=40, rtl=False, padding=0, debug=False, adaptive=False, magiv2=False, magiv2_model=None):
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Error: Image '{image_path}' not found or invalid format.")
        panels = []
        if magiv2 and magiv2_model is not None:
            # Use Magiv2 for panel detection
            from PIL import Image
            # NOTE: Check Magiv2 repo for input requirements (size, color, dtype, etc.)
            with open(image_path, "rb") as file:
                pil_img = Image.open(file).convert("L").convert("RGB")
                np_img = np.array(pil_img)
            character_bank = {"images": [], "names": []}  # Always pass an empty character_bank
            with torch.no_grad():
                results = magiv2_model.do_chapter_wide_prediction([np_img], character_bank=character_bank, use_tqdm=False, do_ocr=False)
            print(f"[MAGIV2 DEBUG] Raw results for {image_path}: {results}")
            if (
                results is not None and
                isinstance(results, list) and
                len(results) > 0 and
                results[0] is not None and
                isinstance(results[0], dict) and
                'panels' in results[0] and
                results[0]['panels'] is not None
            ):
                panel_boxes = results[0]['panels']
                for box in panel_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    panels.append((x, y, w, h))
            else:
                print(f"[ERROR] Magiv2 did not return valid panels for {image_path}. Raw results: {results}\nCheck Magiv2 repo for input requirements and model compatibility.")
                return
        else:
            # Preprocessing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            equalized = cv2.equalizeHist(denoised)
            if adaptive:
                thresh = cv2.adaptiveThreshold(
                    equalized, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
                edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
            else:
                blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
                edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
            kernel = np.ones((5, 5), np.uint8)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            if debug:
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, "debug_gray.png"), gray)
                cv2.imwrite(os.path.join(output_dir, "debug_denoised.png"), denoised)
                cv2.imwrite(os.path.join(output_dir, "debug_equalized.png"), equalized)
                if adaptive:
                    cv2.imwrite(os.path.join(output_dir, "debug_thresh.png"), thresh)
                cv2.imwrite(os.path.join(output_dir, "debug_blurred.png"), blurred)
                cv2.imwrite(os.path.join(output_dir, "debug_edges.png"), edges)
                cv2.imwrite(os.path.join(output_dir, "debug_closed.png"), closed)
            contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            print(f"[DEBUG] {image_path}: Found {len(contours)} contours before filtering.")
            img_area = img.shape[0] * img.shape[1]
            min_area = (min_panel_area_percent / 100) * img_area
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    x, y, w, h = cv2.boundingRect(approx)
                    area = w * h
                    if area > min_area and (w > 50 and h > 50):
                        panels.append((x, y, w, h))
            print(f"[DEBUG] {image_path}: {len(panels)} panels after filtering.")
        # Sort panels by reading order with improved row grouping
        def group_rows(panels, y_tol=30):  # Increased tolerance for better grouping
            rows = []
            for panel in sorted(panels, key=lambda b: b[1]):
                placed = False
                for row in rows:
                    if abs(row[0][1] - panel[1]) < y_tol:
                        row.append(panel)
                        placed = True
                        break
                if not placed:
                    rows.append([panel])
            return rows
        rows = group_rows(panels)
        # Sort rows by the minimum y of the panels in each row (top to bottom)
        rows.sort(key=lambda row: min(p[1] for p in row))
        sorted_panels = []
        for row in rows:
            if rtl:
                row.sort(key=lambda b: -b[0])  # right-to-left
            else:
                row.sort(key=lambda b: b[0])   # left-to-right
            sorted_panels.extend(row)
        panels = sorted_panels
        os.makedirs(output_dir, exist_ok=True)
        metadata = []
        img_h, img_w = img.shape[:2]
        for i, (x, y, w, h) in enumerate(panels):
            x_pad = max(x - padding, 0)
            y_pad = max(y - padding, 0)
            w_pad = min(w + 2 * padding, img_w - x_pad)
            h_pad = min(h + 2 * padding, img_h - y_pad)
            panel_img = img[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            filename = f"panel_{i+1:03d}.png"
            cv2.imwrite(os.path.join(output_dir, filename), panel_img)
            metadata.append({"file": filename, "x": x_pad, "y": y_pad, "w": w_pad, "h": h_pad})
        with open(os.path.join(output_dir, "panels.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Split {len(panels)} panels into: {output_dir}")
    except Exception as e:
        print(f"Error processing '{image_path}': {e}")

def process_images(input_path, output_base_dir, min_panel_area_percent=2.0, canny_threshold1=10, canny_threshold2=40, rtl=False, padding=0, debug=False, adaptive=False, magiv2=False):
    magiv2_model = None
    temp_dir = None
    archive_mode = None
    extracted_dir = None
    # If input_path is a directory, scan for .cbz/.cbr files
    if os.path.isdir(input_path):
        archive_files = sorted(glob.glob(os.path.join(input_path, '*.cbz')) +
                               glob.glob(os.path.join(input_path, '*.cbr')))
        if archive_files:
            print(f"[DEBUG] Found {len(archive_files)} archive(s) in directory: {input_path}")
            for archive_file in archive_files:
                print(f"[DEBUG] Processing archive: {archive_file}")
                # Create subfolder in output directory based on archive name (without extension)
                archive_name = os.path.splitext(os.path.basename(archive_file))[0]
                archive_output_dir = os.path.join(output_base_dir, archive_name)
                process_images(archive_file, archive_output_dir, min_panel_area_percent, canny_threshold1, canny_threshold2, rtl, padding, debug, adaptive, magiv2)
            return
    # Archive extraction logic (single file)
    if input_path.lower().endswith('.cbz'):
        archive_mode = 'zip'
    elif input_path.lower().endswith('.cbr'):
        archive_mode = 'rar'
    if archive_mode:
        temp_dir = tempfile.mkdtemp(prefix='manga_panels_')
        extracted_dir = temp_dir
        try:
            if archive_mode == 'zip':
                with zipfile.ZipFile(input_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            elif archive_mode == 'rar':
                if rarfile is None:
                    print("rarfile module is not installed. Please install it to process .cbr files.")
                    return
                if not rarfile.is_rarfile(input_path):
                    print(f"File {input_path} is not a valid RAR archive.")
                    return
                try:
                    with rarfile.RarFile(input_path) as rar_ref:
                        rar_ref.extractall(temp_dir)
                except rarfile.RarCannotExec as e:
                    print("'unrar' or 'rar' is not installed or not in PATH. Please install it to process .cbr files.")
                    return
        except Exception as e:
            print(f"Failed to extract archive {input_path}: {e}")
            if temp_dir:
                shutil.rmtree(temp_dir)
            return
        input_path = temp_dir
    try:
        if magiv2:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            magiv2_model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True).to(device).eval()
        if os.path.isdir(input_path):
            print(f"[DEBUG] Searching for images in: {input_path}")
            image_files = sorted(glob.glob(os.path.join(input_path, '**', '*.jpg'), recursive=True) +
                                 glob.glob(os.path.join(input_path, '**', '*.jpeg'), recursive=True) +
                                 glob.glob(os.path.join(input_path, '**', '*.png'), recursive=True))
            print(f"[DEBUG] Found {len(image_files)} images.")
            if not image_files:
                print(f"No images found in directory: {input_path}")
                return
            for image_file in tqdm(image_files, desc="Processing images"):
                image_stem = os.path.splitext(os.path.basename(image_file))[0]
                output_dir = os.path.join(output_base_dir, image_stem)
                split_panels(
                    image_file,
                    output_dir,
                    min_panel_area_percent=min_panel_area_percent,
                    canny_threshold1=canny_threshold1,
                    canny_threshold2=canny_threshold2,
                    rtl=rtl,
                    padding=padding,
                    debug=debug,
                    adaptive=adaptive,
                    magiv2=magiv2,
                    magiv2_model=magiv2_model if magiv2 else None
                )
        else:
            if not os.path.isfile(input_path):
                print(f"Input file not found: {input_path}")
                return
            image_stem = os.path.splitext(os.path.basename(input_path))[0]
            output_dir = os.path.join(output_base_dir, image_stem)
            split_panels(
                input_path,
                output_dir,
                min_panel_area_percent=min_panel_area_percent,
                canny_threshold1=canny_threshold1,
                canny_threshold2=canny_threshold2,
                rtl=rtl,
                padding=padding,
                debug=debug,
                adaptive=adaptive,
                magiv2=magiv2,
                magiv2_model=magiv2_model if magiv2 else None
            )
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="Split comic pages into panels.")
    parser.add_argument("input", help="Input image path or directory")
    parser.add_argument("-o", "--output", help="Output directory (default: output/)")
    parser.add_argument("--min_area", type=float, default=2.0, 
                       help="Minimum panel area as %% of total image (default: 2%%)")
    parser.add_argument("--canny1", type=int, default=10, 
                       help="Canny edge detection lower threshold (default: 10)")
    parser.add_argument("--canny2", type=int, default=40, 
                       help="Canny edge detection upper threshold (default: 40)")
    parser.add_argument("--rtl", action="store_true", help="Sort panels for right-to-left reading order (manga style)")
    parser.add_argument("--padding", type=int, default=0, help="Padding (in pixels) to add around each panel (default: 0)")
    parser.add_argument("--debug", action="store_true", help="Save intermediate images for debugging")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive thresholding before Canny edge detection")
    parser.add_argument("--magiv2", action="store_true", help="Use Magiv2 AI model for panel detection instead of OpenCV")
    args = parser.parse_args()
    if not args.output:
        args.output = os.path.join("output")
    process_images(
        args.input,
        args.output,
        min_panel_area_percent=args.min_area,
        canny_threshold1=args.canny1,
        canny_threshold2=args.canny2,
        rtl=args.rtl,
        padding=args.padding,
        debug=args.debug,
        adaptive=args.adaptive,
        magiv2=args.magiv2
    )

if __name__ == "__main__":
    main()