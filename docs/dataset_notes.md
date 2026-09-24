
# Dataset Preparation

## Project
Worker Safety Helmet Detection using YOLOv8

## Target Classes
- person
- helmet
- no_helmet

## 1. Manual Dataset

Images were collected from public image sources like Kaggle datasets and Pexels and manually annotated using Roboflow.

- 121 images
- 544 bounding-box annotations
- Classes: person, helmet, no_helmet

Bounding boxes were drawn around visible people, safety helmets, and uncovered upper-head regions.

## 2. Public Dataset

Additional annotated images were collected from public datasets on Roboflow Universe. Images were selected based on their relevance to workplace safety, annotation quality, and diversity.

Class names were standardized, irrelevant classes such as safety vests were removed, and the cleaned images were combined with the manual dataset.

- Public source images selected: 237
- Additional public images were also reviewed and merged.
- Duplicate images were excluded during merging.

### Sources

1. [Worker-Safety](https://universe.roboflow.com/computer-vision/worker-safety)
2. [PPE (No Personal)](https://universe.roboflow.com/asimov/ppe-no-personal)

## Final Dataset

After cleaning and merging, the final dataset contains:

- Total images: 316
- Total annotations: 1,785
- Classes: 3

### Class Distribution

- person: 872
- helmet: 545
- no_helmet: 368

### Dataset Split

- Training: 221 images (70%)
- Validation: 63 images (20%)
- Testing: 32 images (10%)

The final dataset was exported from Roboflow in YOLOv8 format for training in Google Colab.