# Deep Learning-Based Lung Cancer Detection and Classification

This repository presents a deep learning approach for detecting and classifying lung cancer using CT scan images. The project employs pre-trained CNN architectures (VGG16 and VGG19) with transfer learning to automate and optimize the diagnostic process.

## Project Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early and accurate diagnosis is essential for effective treatment. Traditional diagnostic methods are invasive and prone to human error. This project introduces a non-invasive, AI-powered system that classifies lung cancer subtypes with high accuracy using medical imaging.

## Objectives

- Automate lung cancer detection using CT images.
- Reduce dependency on invasive diagnostic procedures and expert interpretation.
- Enhance diagnostic accuracy using deep learning models (VGG16, VGG19).
- Enable early diagnosis to support timely treatment planning.

## Dataset

- CT scan images categorized into:
  - Adenocarcinoma
  - Large Cell Carcinoma
  - Squamous Cell Carcinoma
  - Normal
- Images are preprocessed: normalized, resized, noise-removed, and augmented.
- Dataset used is publicly available and included in this repository.

## Methodology

1. **Data Preprocessing**  
   - Normalization, resizing, denoising, and augmentation.

2. **Model Selection & Transfer Learning**  
   - Use of VGG16 and VGG19 as base models.
   - Custom classification layers added.
   - Fine-tuned top layers to adapt for lung cancer subtypes.

3. **Training & Evaluation**  
   - Trained on preprocessed images.
   - Evaluated using accuracy, precision, recall, and F1-score.
   - Visualized using confusion matrix and classification report.

4. **Real-Time Prediction**  
   - Image-based predictions with class labels and confidence scores.

## Results

- Achieved over 86.5% accuracy.
- VGG19 showed slightly better performance, especially in classifying large cell carcinoma.
- Visual results include prediction samples, confusion matrices, and performance plots.

## Future Scope

- Integration with hospital systems for real-time diagnostics.
- Extension to multi-modal models using genomics/proteomics.
- Deployment in mobile/web apps for remote diagnostics.
- Clinical validation on larger and more diverse datasets.

## Requirements

- Python 3.x  
- TensorFlow / Keras  
- NumPy, Matplotlib, Scikit-learn  
- Jupyter Notebook / Google Colab

## License

This project is licensed under the MIT License.
