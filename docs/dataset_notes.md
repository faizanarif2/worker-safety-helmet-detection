# Dataset Preparation

## Project

Worker Safety Helmet Detection using YOLOv8

## Target Classes

- person
- helmet
- no_helmet

## Dataset Strategy

The final dataset will combine:

- manually annotated workplace/construction images
- carefully selected public annotated images

The public images will be reviewed before use to ensure that the annotations and visual domain are suitable for this project.

## Manual Dataset

### Annotation Tool

Roboflow Annotate

### Manual Annotation Progress

- Images manually annotated: 110+
- Annotation type: Bounding boxes
- Annotation status: Completed

### Annotation Guidelines

- `person`: bounding box around the whole visible person
- `helmet`: tight bounding box around the visible safety hard hat
- `no_helmet`: tight bounding box around the visible uncovered upper-head region
- Partially visible people are annotated when clearly identifiable
- Objects are skipped when the class cannot be determined confidently
- Each visible target object receives its own bounding box

### Dataset Diversity

The manually selected images include variation in:

- worker count
- camera angle
- background
- lighting
- object size
- helmet and no-helmet cases
- partially visible workers
- crowded workplace scenes

## Public Annotated Data

A supplementary public object-detection dataset will also be used.

The public images will be:

- manually reviewed before inclusion
- selected for relevance to workplace/construction safety
- checked for annotation quality
- checked for duplicates and near-duplicate video frames
- mapped to the project classes where necessary

The final public dataset source and license will be documented after the final source is selected.

## Final Dataset Preparation

After combining the manually annotated and reviewed public images, the dataset will be prepared in YOLO format and split into:

- 70% training
- 20% validation
- 10% testing