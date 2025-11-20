# A-VLM-AD-Baseline: Zero-Shot Anomaly Detection for Brain MRI using CLIP

This project implements a baseline approach for Zero-Shot Anomaly Detection on a Brain MRI dataset using the pre-trained CLIP (Contrastive Language-Image Pre-training) visual encoder. The method relies on extracting deep, generalized features from the images, calculating a "normal" reference vector, and classifying test images based on their Cosine Distance to this reference.

## 🚀 Getting Started

**Prerequisites**
This code is designed to be run in a Python environment (like a Jupyter Notebook or Google Colab) and requires the following libraries:
!pip install --quiet kagglehub
!pip install --quiet open_clip_torch
!pip install --quiet torch torchvision matplotlib scikit-learn
## Dataset
The project automatically downloads the necessary data using the kagglehub library.

* Dataset: navoneel/brain-mri-images-for-brain-tumor-detection

* Content: MRI images split into no (Normal) and yes (Tumor).

## 🧠 Methodology: Cosine Distance Anomaly Score
1. **Feature Encoding**: The pre-trained CLIP visual encoder (ViT-B-32) is used to generate a high-dimensional feature vector (embedding) for every image.
2. **Reference Vector Creation**: All feature vectors from the Normal (no) class are averaged to create a single, representative Reference Vector. This vector encapsulates the visual properties of a "healthy" brain.
3. **Anomaly Scoring**: For all test images (Normal and Tumor), the Cosine Similarity is calculated between its feature vector and the Reference Vector.
4. **Anomaly Distance**: The Anomaly Score is defined as the Cosine Distance (Distance = 1 - Similarity).
   * **Low Distance** (High Similarity) indicates the image is close to the normal reference (predicted Normal).
   * **High Distance** (Low Similarity) indicates the image is an outlier (predicted Anomaly/Tumor).




## 📊 Expected Output

Upon execution, the script will output the download status, the count of images, and the status of feature extraction. The final output is a histogram plot visualizing the distribution of the anomaly scores:

* The Normal samples should be clustered on the left side (closer to 0 distance).
* The Tumor samples should be shifted towards the right side (higher distance), demonstrating that the model can separate anomalies from normal images based on the CLIP feature space.
