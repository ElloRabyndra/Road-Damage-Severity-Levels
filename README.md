# Road Damage Severity Levels

## Project Overview
This project presents an end-to-end machine learning pipeline for detecting road damages and classifying their severity levels. Utilizing the RDD2022 dataset, the solution is designed to assist in infrastructure maintenance by automatically identifying the type of road damage and evaluating its critical state based on visual area footprint.

The system employs a two-stage approach:
1. **Object Detection**: A YOLOv8m model detects the location and type of road damage.
2. **Severity Classification**: A Convolutional Neural Network (CNN) evaluates the detected regions to classify the severity level (Low, Medium, High).

## Severity Criteria
The severity classification is fundamentally based on the area ratio of the damage relative to the entire image frame:
- **Low (Rendah)**: Damage covers less than 5% of the total frame. Examples include thin cracks or small potholes.
- **Medium (Sedang)**: Damage covers between 5% and 15% of the total frame. Examples include noticeable cracks or medium-sized potholes.
- **High (Tinggi)**: Damage covers more than 15% of the total frame. Examples include wide alligator cracks or large potholes.

## Dataset
The model is trained on a processed version of the RDD2022 dataset. The dataset includes the following damage classes:
- Longitudinal Crack (Class 0)
- Transverse Crack (Class 1)
- Alligator Crack (Class 2)
- Other Corruption (Class 3)
- Pothole (Class 4)

## Project Structure
The repository is structured around a series of sequential Jupyter Notebooks that document the entire workflow:

- `notebooks/1_dataset_analysis.ipynb`: Initial exploration, statistical analysis, and visualization of the raw dataset.
- `notebooks/2_balanced_sampling_severity.ipynb`: Techniques for handling class imbalances and generating balanced samples for severity modeling.
- `notebooks/3_manual_validation_helper.ipynb`: Tooling for manual validation of automatically calculated severity thresholds against human visual assessment.
- `notebooks/4_Road_Severity_Yolo_training.ipynb`: Training and validation of the YOLOv8m object detection model.
- `notebooks/5_CNN_Severity_Classification.ipynb`: Training of the CNN model for classifying the severity of cropped damage regions.
- `notebooks/6_End_to_End_Inference_Evaluation.ipynb`: The final pipeline integrating YOLOv8m and CNN models for end-to-end inference and performance evaluation.

## Performance Metrics
### YOLOv8m Object Detection
- **mAP@0.5**: 0.530
- **mAP@0.5:0.95**: 0.260
- **Precision**: 0.581
- **Recall**: 0.505

### CNN Severity Classification
- **Overall Accuracy**: 76.9%
- **F1-Score (Macro Avg)**: 0.598
- **Low Severity (Rendah)**: Precision 0.82, Recall 0.92
- **Medium Severity (Sedang)**: Precision 0.50, Recall 0.27
- **High Severity (Tinggi)**: Precision 0.58, Recall 0.57

## Results and Visualizations
The `reports/` directory contains evaluation metrics in JSON format and comprehensive visualizations in the `reports/figure/` directory:
- YOLOv8 Training Visualizations: `reports/figure/yolo/training_visualization.png`
- YOLOv8 Sample Inference: `reports/figure/yolo/sample_inference.png`
- CNN Training Curves: `reports/figure/cnn/training_curves.png`
- CNN Sample Predictions: `reports/figure/cnn/sample_predictions.png`

## Integrated Pipeline Evaluation
The `inference/` directory contains the final evaluation results from running the integrated end-to-end pipeline (YOLOv8 followed by CNN) on the test set. It includes:
- `evaluation_summary.txt`: A concise text summary of the integrated pipeline's performance across 5,758 test images.
- `inference_results.json`: Detailed, per-image detection and classification results.
- `final_evaluation_report.json`: Comprehensive pipeline evaluation metrics.
- `cnn_severity_metrics.json` & `yolo_detection_metrics.json`: Specific model metrics generated during the inference phase.
- `figure/`: Final visualizations, including inference outputs (`inference_visualitation.png`) and summary charts (`summary_charts.png`).

## Validation Process
A crucial part of this project involves ensuring the automated severity calculation aligns with human perception. The `validation/` directory contains instructions (`VALIDATION_INSTRUCTIONS.txt`) for human annotators to manually verify the bounding box area thresholds against visual guidelines.
