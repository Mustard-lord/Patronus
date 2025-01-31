# PATRONUS: Safeguarding Text-to-Image Models against White-Box Adversaries
>Note:The code and documentation will be updated and finalized upon the publication of the paper. Thank you for your patience and understanding.
## Env Setup
```bash
conda env create -f environment.yml
```
## Datasets
### Preparing Data
Unsafe Data:[NSFW Dataset](https://github.com/alex000kim/nsfw_data_scraper)

Safe Data: MS COCO Caption Dataset & ImageNet-1k
### Preprocess Data
#### Feature Extraction
```bash
python tools/extract_feature_from_image.py
```
#### LLaVA Prompt Depiction
>LLaVA-13B is used to caption NSFW-56K
```bash
python tools/caption.py
```
## Decoder Process
### Train
```bash
bash scripts/decoder/unlearn.sh
bash scripts/decoder/final.sh
```
### Test
```bash 
bash scripts/decoder/eval.sh
bash scripts/decoder/overall_performance.sh
```
## Diffusion Process
### Train
>Most of the settings are included in the 'yaml' files in the 'config' folder
```bash
bash scripts/diffusion/unlearn.sh
bash scripts/diffusion/anti_finetune.sh
```
### Test
```bash 
bash scripts/diffusion/FTAT.sh
```