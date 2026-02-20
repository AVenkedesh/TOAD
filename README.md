# TOAD (Dream Team Engineering)

TOAD is a deep learning pipeline for histology segmentation in experimental knee osteoarthritis (OA).  
This repo should contain **code only** (no datasets, no model weights).

## Requirements
- Conda (Miniforge/Anaconda)
- Python 3.10 recommended

Create the environment:
```bash
conda env create -f environment.yml
conda activate seko

## Running Inference (Single Image Forward Pass)

You can run a trained SEKO model on a single image and automatically save:

1. The raw predicted mask (class IDs)
2. A grayscale visualization of the mask
3. An overlay of the mask on the original image

### Basic Usage

From the project root:

```bash
python src/forward_pass.py \
  --model "/path/to/model.keras" \
  --image "/path/to/input_image.tif" \
  --output "/path/to/output_mask.png"