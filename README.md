This repository presents an intelligent diagnostic system powered by *Convolutional Neural Networks (CNNs), aimed at early **lung cancer detection and classification* using *CT scan images. Built on **VGG16 and VGG19 architectures, the models leverage **transfer learning, **image preprocessing, and **real-time evaluation tools* to assist radiologists in accurate diagnosis.

---

### *Problem Statement*

Manual analysis of CT scans is time-consuming and prone to human error, especially in early cancer stages. Traditional methods lack consistency, are resource-intensive, and often result in late diagnosis. This project addresses these gaps by automating classification using deep learning.

---

### *Objectives*

- Accurately classify CT images into four categories:  
  Adenocarcinoma, Squamous Cell Carcinoma, Large Cell Carcinoma, and Normal
- Improve diagnosis reliability with automated segmentation and classification
- Ensure real-time performance even on CPU systems (Intel Core i5+)
- Provide interpretable results via confidence scores and visual outputs

---

### *Dataset Details*

- *Source*: Kaggle Lung Cancer CT Scan Dataset  
- *Format*: PNG images, RGBA color mode  
- *Resolution*: Original – 381×282 px; resized to 224×224 px for model compatibility  
- *Split*:  
  - 70% – Training  
  - 8% – Validation  
  - 22% – Testing  
- *Class Labels*: Encoded as folder names; include metadata (e.g., tumor stage, location)

---

### *Preprocessing Pipeline*

1. *Resizing* to 224×224  
2. *Normalization* of pixel values to [0,1]  
3. *Contrast enhancement* using Histogram Equalization / CLAHE  
4. *Noise reduction* using Gaussian, Median, and Wiener filters  
5. *Lung segmentation* to isolate relevant ROI and reduce background interference  

---

### *Model Architecture*

- *Base Models*: VGG16 & VGG19 (pre-trained on ImageNet)  
- *Custom Layers*:
  - GlobalAveragePooling2D
  - Dense(512, activation='relu')
  - Dropout(0.4)
  - Dense(4, activation='softmax')  

- *Optimizer*: Adam  
- *Loss Function*: Categorical Crossentropy  
- *Learning Rate Scheduler, **Early Stopping, and **Model Checkpoints* implemented for stability and optimal training.

---

### *Environment Setup*

- *Platform*: Google Colab  
- *Hardware*: NVIDIA Tesla T4 GPU (16 GB VRAM)  
- *Tools/Libraries*: TensorFlow, Keras, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  

---

### *Implementation Flow*

1. *Load and preprocess data* using ImageDataGenerator  
2. *Initialize CNN model* with pre-trained base  
3. *Train and fine-tune* model with validation monitoring  
4. *Evaluate* using accuracy, precision, recall, F1-score  
5. *Visualize* with confusion matrices and metric bar graphs  
6. *Predict* real-time test images with GUI-style output including:
   - Predicted class
   - Malignant/Benign label
   - Confidence score
   - Visual segmentation overlay

---

### *Performance Highlights*

#### *VGG16:*
- Accuracy: ~87%  
- Precision: up to 1.00 (Normal)  
- Recall: Highest for Adenocarcinoma (0.87)  
- Balanced F1-scores across all classes

#### *VGG19:*
- Accuracy: ~86%  
- Perfect Recall for Large Cell Carcinoma  
- Slightly better precision in some cancer subtypes

---

### *Visual Output Samples*

- Confusion matrices per model  
- Bar charts: Class-wise precision, recall, F1-score  
- Image predictions with overlaid class and score  
- Side-by-side performance comparisons (VGG16 vs VGG19)

---

### *Project Scope*

- CT scan-based classification only  
- Modular design allows future integration with:
  - Histopathology images
  - PET scans
  - Genomic data  
- Does not replace clinical diagnosis—intended as a *decision support tool*

---

### *Limitations*

- Requires labeled CT data  
- Some misclassifications due to class imbalance  
- ROI segmentation is rule-based, not learned

---

This project shows how *deep learning combined with image processing* can enhance diagnostic accuracy in lung cancer detection—making it faster, more consistent, and scalable across healthcare systems.
