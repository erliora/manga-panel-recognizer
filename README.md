# Manga Panel Splitter

A Python tool to automatically split manga/comic pages into individual panels for better display on e-readers (e.g., Kindle KFX/KPF formats). Uses OpenCV for robust panel detection and supports batch processing, debug output, and flexible reading order. Now supports AI-based panel detection with Magiv2.

## Features
- **Automatic panel detection** using edge detection and contour analysis (OpenCV)
- **AI-based panel detection** with [MagiV2](https://github.com/ragavsachdeva/magi) for more robust results
- **Batch processing**: process a single image or an entire directory
- **Right-to-left (RTL) and left-to-right (LTR) reading order**
- **Panel padding**: add a margin around each panel
- **Metadata export**: JSON file with panel coordinates and filenames
- **Debug mode**: save intermediate images for troubleshooting
- **Robust preprocessing**: denoising and contrast adjustment

## Installation

1. Install Python 3.8+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # For Magiv2 support:
   pip install torch torchvision transformers pillow tqdm opencv-python
   ```
   - On Apple Silicon (M1/M2/M3), PyTorch will use MPS (Metal) backend. On Nvidia GPUs, CUDA is used if available.

## Usage

### OpenCV Panel Detection (Default)
```bash
python panel_splitter.py input/ --output output/
```

### Magiv2 AI Panel Detection
```bash
python panel_splitter.py input/ --output output/ --magiv2
```
- This uses the [MagiV2 model](https://huggingface.co/ragavsachdeva/magiv2) for panel detection.
- **Note:** Magiv2 requires an empty `character_bank` dictionary to be passed (handled automatically by the script).
- Magiv2 panel coordinates are floats and are rounded for cropping.
- Device selection is automatic: MPS (Mac), CUDA (Nvidia), or CPU.

### Other Options
- `--rtl` : Right-to-left reading order (manga style)
- `--padding N` : Add N pixels of margin around each panel
- `--debug` : Save intermediate images for troubleshooting
- `--adaptive` : Use adaptive thresholding (OpenCV only)
- `--min_area` : Minimum panel area as % of image
- `--canny1`, `--canny2` : Canny edge detection thresholds (OpenCV only)

## Example
```bash
python panel_splitter.py input/ --output output/ --magiv2 --rtl --padding 10
```

## Troubleshooting
- **Magiv2 returns no panels or None:**
  - Ensure you are using the latest version of the script and dependencies.
  - The script must pass an empty `character_bank` dictionary to Magiv2 (handled automatically).
  - If you see `[MAGIV2 DEBUG]` output, check the structure of the results. If results are `None`, check your PyTorch and Transformers versions.
  - On Apple Silicon, PyTorch will use MPS. If you see device warnings, try updating PyTorch.
- **Panel crops are off:**
  - Magiv2 returns float coordinates; the script rounds them for cropping.
- **OpenCV detects too few/many panels:**
  - Tune `--canny1`, `--canny2`, and `--min_area`.

## Credits
- [MagiV2](https://github.com/ragavsachdeva/magi) by Ragav Sachdeva and Gyungin Shin and Andrew Zisserman
- OpenCV, PyTorch, Transformers, PIL
