
# VTON4Everyone: Virtual Try-On Pipelines

## Overview

VTON4Everyone is a repository containing multiple Virtual Try-On (VTON) pipelines, each designed to perform virtual try-on tasks using different models or approaches. Currently, the repository includes the `vton_sd15.py` script, which leverages Stable Diffusion + ControlNet + Inpaint + IP adapter for VTON.

## Prerequisites

Ensure you have the following prerequisites installed:

- Python 3

## Installation

Clone the repository:

```bash
git clone https://github.com/Flode-Labs/VTON4Everyone.git
cd VTON4Everyone
```

Install the required packages:

```bash
pip install git+https://github.com/huggingface/diffusers
pip install -q transformers accelerate controlnet-aux
```

## Usage

### VTON-SD15 (Stable Diffusion + ControlNet + Inpaint + IP Adapter)

Set the following variables to the paths of the images you want to use in the `vton_sd15.py` script:


```python
    init_image_path = load_image("./test/model.png")
    mask_image = load_image("./test/mask.png")
    ip_image = load_image("./test/garment.png") 
    prompt = "a woman wearing a dress, best quality, high quality"
```

Note: The mask image should be a black and white image with the same dimensions as the init image. The white pixels in the mask image will be replaced. The head, hands and feet should be black in the mask image. Currently we're working on a script to automatically generate the mask image from the init image.

Then run the script:
```bash
python sd1_5.py
```

The script will process the inputs and generate an output image of the model wearing the specified garment.

### Adding New Pipelines

We're actively working on expanding our VTON offerings. Stay tuned for additional pipelines. 


## Notes

- This pipelines are experimental and may not work as expected in all cases.
- Ensure that the image paths provided are valid and accessible.
- Experiment with different prompts and params to achieve varied and desired results.
- Contributions to additional VTON pipelines are welcome. Feel free to create a new pipeline and submit a pull request.

## TODO

- Add a script to automatically generate the mask image from the init image.
- Add additional VTON pipelines.
- Improve inpainting quality(maybe use a different inpainting model or refining the image).

## Acknowledgments

- VTON4Everyone is a collaborative effort, and we appreciate contributions from the community.
- Special thanks to the contributors of the underlying libraries and models.

