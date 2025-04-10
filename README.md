# Brain-Tumor-Detection-Using-Deep-Learning

![Project-Cover-Photo](https://github.com/user-attachments/assets/9cb92e1d-c67f-4f9d-90c5-2f2e65064ce4)

Introduction - 

The occurrence of brain tumor patients in India is steadily rising, more and more cases of brain tumors are reported each year in India across varied age groups. The International Association of Cancer Registries (IARC) reported that there are over 28,000 cases of brain tumours reported in India each year and more than 24,000 people reportedly die due to brain tumours i.e 85.7% people die annually from the total reported cases. Brain tumors are a serious condition and in most cases fatal in later stages if not detected early on.

Healthcare sector can benefit significantly from the field of Artificial Intelligence by developing systems which have the capability to detect these fatal diseases in the early stages because most diseases when detected early can be treated successfully before it's too late and same is the case with various different kinds of cancer.

The goal of the project was to develop a deep learning model which has the capability to classify the brain MRI images consisting of tumors with higher accuracy.






About Dataset -

Refer to README.md file in the Brain Tumor Dataset directory in this repository to get a clear idea about the dataset and the preprocessing steps.
The below image gives a glimpse about the different kinds of tumors with its localisation through a binary map after pre-processing the .mat file in which the image data was stored.
![Brain-Tumor-MRI-With-Localisation-Masks](https://github.com/user-attachments/assets/9e6f7283-3090-4df5-80cc-ade1a4302ae6)

Brain-MRI-Images-With-Localisation-Masks








Results - 

Developed 3 Deep Neural Network models i.e. Multi-Layer Perceptron, AlexNet-CNN, and Inception-V3 in order to classify the Brain MRI Images to 4 different independent classes.
Inception-V3 model used is a pre-trained on the ImageNet dataset which consist of 1K classes but for this project we have tuned the later part i.e. the Fully-Connected part of the model while retaining the weights of the CNN part to satisfy the needs of this work.
![Screenshot 2025-04-10 175212](https://github.com/user-attachments/assets/7f5d9550-d5ec-46ef-bb6f-0865c7655cb5)

The pre-trained Inception-V3 model has performed significantly well with an accuracy of 82.57% as compare to AlexNet-CNN and Multi-Layer Perceptron deep neural network model.


