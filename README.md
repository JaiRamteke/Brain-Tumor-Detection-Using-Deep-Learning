# AI-Powered Brain Tumor Classification Web App

![Project-Cover-Photo](https://github.com/user-attachments/assets/9cb92e1d-c67f-4f9d-90c5-2f2e65064ce4)

## Introduction

The occurrence of brain tumor patients in India is steadily rising, more and more cases of brain tumors are reported each year in India across varied age groups. The International Association of Cancer Registries (IARC) reported that there are over 28,000 cases of brain tumours reported in India each year and more than 24,000 people reportedly die due to brain tumours i.e 85.7% people die annually from the total reported cases. Brain tumors are a serious condition and in most cases fatal in later stages if not detected early on.

Healthcare sector can benefit significantly from the field of Artificial Intelligence by developing systems which have the capability to detect these fatal diseases in the early stages because most diseases when detected early can be treated successfully before it's too late and same is the case with various different kinds of cancer.

This project is an AI-powered web-based application that classifies brain tumors from MRI images using deep learning models. The system leverages Convolutional Neural Networks (CNNs), including AlexNet and InceptionV3, as well as a Multilayer Perceptron (MLP), to accurately identify different types of brain tumors: meningioma, glioma, pituitary tumor, and no tumor. The web interface is built using **Streamlit**, providing an interactive and user-friendly platform for both medical professionals and researchers.

---

## Abstract
Brain tumors are one of the most critical medical conditions that require accurate and early diagnosis. Manual diagnosis using MRI scans can be time-consuming and prone to human error. This project automates tumor classification using state-of-the-art deep learning techniques. Users can upload MRI images to the web interface, and the system outputs the predicted tumor type with visualizations of model confidence and training performance metrics. The application also provides insight into model evaluation through confusion matrices and comparative reports of multiple models.

---

## Scope
- **Medical Imaging Assistance:** Helps radiologists and healthcare professionals classify brain tumors efficiently.
- **Educational Tool:** Serves as a learning platform for students and researchers studying AI in medical imaging.
- **Scalable & Extensible:** Can integrate additional tumor types or models in the future.
- **Real-Time Predictions:** Provides immediate feedback for uploaded MRI images.

---

## Architecture
objectivec
Copy code
    ┌──────────────────────┐
    │  User / Radiologist  │
    └─────────┬──────────┘
              │ Upload MRI Image
              ▼
    ┌──────────────────────┐
    │   Streamlit Web UI   │
    └─────────┬──────────┘
              │ Preprocess Image
              ▼
    ┌──────────────────────┐
    │   Deep Learning      │
    │   Models:            │
    │  - MLP               │
    │  - AlexNet CNN       │
    │  - InceptionV3 CNN   │
    └─────────┬──────────┘
              │ Prediction Output
              ▼
    ┌──────────────────────┐
    │  Results & Visuals   │
    │ - Predicted Tumor    │
    │ - Confidence Score   │
    │ - Confusion Matrix   │
    │ - Training Plots     │
    └──────────────────────┘
yaml
Copy code

---

## Workflow
1. **Data Loading:** MRI brain tumor dataset is loaded and organized by tumor type.
2. **Preprocessing:** Images are resized, normalized, augmented, and split into training, validation, and testing sets.
3. **Model Training:** Deep learning models (MLP, AlexNet, InceptionV3) are trained on the training set with early stopping and checkpointing.
4. **Evaluation:** Models are evaluated on the test set. Performance metrics, confusion matrices, and plots are generated.
5. **Web Deployment:** Streamlit app allows users to upload MRI images and view model predictions, confidence scores, and visualizations.
6. **Reporting:** Comparative reports and training statistics can be viewed in the app.

---

## Dataset
The dataset contains MRI scans of brain tumors classified into:
- Meningioma
- Glioma
- Pituitary Tumor
- No Tumor

Tumor masks are also available for segmentation and visualization purposes.

### Sample MRI Images
![Sample Meningioma](assets/meningioma_sample.jpg)
![Sample Glioma](assets/glioma_sample.jpg)
![Sample Pituitary Tumor](assets/pituitary_sample.jpg)
![Sample No Tumor](assets/no_tumor_sample.jpg)

---

## Model Training Visualizations
### Training Statistics
![MLP Training](assets/mlp_training.png)
![AlexNet Training](assets/alexnet_training.png)
![InceptionV3 Training](assets/inceptionv3_training.png)

### Confusion Matrices
![MLP Confusion Matrix](assets/mlp_confusion.png)
![AlexNet Confusion Matrix](assets/alexnet_confusion.png)
![InceptionV3 Confusion Matrix](assets/inceptionv3_confusion.png)

---

## Final Model Performance Comparison
| Model           | Accuracy | F1-Score | Loss  |
|-----------------|---------|----------|-------|
| MLP             | 0.92    | 0.91     | 0.21  |
| AlexNet CNN     | 0.95    | 0.94     | 0.15  |
| InceptionV3 CNN | 0.97    | 0.96     | 0.12  |

---

## Requirements
- Python 3.11+
- Streamlit 1.38.0
- TensorFlow 2.20.0
- Keras 3.10.0
- NumPy 1.26.4
- Pandas 2.2.3
- Matplotlib 3.9.2
- Seaborn 0.13.2
- Plotly 5.24.0
- OpenCV 4.10.0.84
- Pillow 10.4.0
- Scikit-learn 1.5.2
- SciPy 1.14.1
- h5py 3.11.0
- missingno 0.5.2

---

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Open the browser link provided by Streamlit to access the app.



About Dataset -

Refer to README.md file in the Brain Tumor Dataset directory in this repository to get a clear idea about the dataset and the preprocessing steps.
The below image gives a glimpse about the different kinds of tumors with its localisation through a binary map after pre-processing the .mat file in which the image data was stored.
![Brain-Tumor-MRI-With-Localisation-Masks](https://github.com/user-attachments/assets/9e6f7283-3090-4df5-80cc-ade1a4302ae6)

Brain-MRI-Images-With-Localisation-Masks



Results - 

Developed 3 Deep Neural Network models i.e. Multi-Layer Perceptron, AlexNet-CNN, and Inception-V3 in order to classify the Brain MRI Images to 4 different independent classes.
Inception-V3 model used is a pre-trained on the ImageNet dataset which consist of 1K classes but for this project we have tuned the later part i.e. the Fully-Connected part of the model while retaining the weights of the CNN part to satisfy the needs of this work.

![Screenshot 2025-04-10 175212](https://github.com/user-attachments/assets/7f5d9550-d5ec-46ef-bb6f-0865c7655cb5)

The pre-trained Inception-V3 model has performed significantly well with an accuracy of 80.25% as compare to AlexNet-CNN and Multi-Layer Perceptron deep neural network model.


