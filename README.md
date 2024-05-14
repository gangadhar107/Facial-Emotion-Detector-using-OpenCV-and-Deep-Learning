# Facial-Emotion-Detector-using-OpenCV-and-Deep-Learning
This project aims to detect 7 emotions (such as Happy, Sad, Anger, Surprise, Disgust, Neutral, Fear.) in faces using OpenCV for face detection and a deep learning model for emotion classification. 
# Dataset Used
The FER 2013 dataset serves as the foundation for this project. You can find this dataset on [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013). Our model leverages this dataset, which contains labeled facial expressions, to recognize different emotions.
# Python Libraries Used
Before diving into the project, ensure that you have the following libraries installed. If you already have them, feel free to skip this step:
```python
pip install tensorflow keras
pip install opencv-python
```
# Approach 
1. Face Detection using Haar Cascades:
- Utilized OpenCV's Haar Cascades for face detection.
- The classifier was loaded using the following code snippet:
```python
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
```
2. Emotion Classification Model:
- The emotion classification model is based on a deep learning architecture using Convolutional Neural Networks(CNNs).
- Prior to feeding the images into the neural network, we need to preprocess them by resizing and normalizing.
# Training and Testing the Model 
1. Training the Model:
- Open the `train emotion detector.ipynb` notebook.
- Inside the notebook, you’ll find code for loading the FER 2013 dataset, preprocessing the images, designing the CNN architecture, and training the model.
- Execute the cells sequentially to train your model.
2. Testing the Model:
- Open the `test emotion detector.ipynb` notebook.
- In this notebook, load the trained model weights (HDF5 file) obtained from the training step.
- Use test images (or video frames) to evaluate the model’s performance.
