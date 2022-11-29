# image-captioning

## Problem statement
The problem introduces a captioning task, which requires a computer vision system to both localize and describe salient regions in images in natural language. The image captioning task generalizes object detection when the descriptions consist of a single word. Given a set of images and prior knowledge about the content find the correct semantic label for the entire image(s).

## Solution proposed
Image caption generator is a process of recognizing the context of an image and annotating it with relevant captions using deep learning, and computer vision. It includes the labeling of an image with English keywords with the help of datasets provided during model training. Flickr30k dataset is used to train the CNN model called Inception v3. Inception v3 is responsible for image feature extraction. These extracted features will be fed to the LSTM model which in turn generates the image caption.

## Dataset used
Flickr30k dataset is been used for this project. Here the model is been trained on 100 images.

## Tech stack used
1. Python 3.8
2. FastAPI
3. Deep learning
4. Computer vision - Inception V3 model
5. Natural language preocessing - LSTM Architecture
6. Docker

## Infrastructure required
1. AWS S3
2. AWS EC2 instance
3. AWS ECR
4. GitHub Actions

## How to run

Step 1. Cloning the repository.
```
git clone https://github.com/Deep-Learning-01/image-captioning.git
```
Step 2. Create a conda environment.
```
conda create -p env python=3.8 -y
```
```
conda activate env/
```
Step 3. Install the requirements
```
pip install -r requirements.txt
```
Step 4. Export the environment variables
```
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>
```
Step 5. Run the application server
```
python app.py
```
Step 6. Train application
```
http://localhost:8000/train
```
Step 7. Prediction application
```
http://localhost:8000/predict
```

## Run locally
1. Check if the Dockerfile is available in the project directory.
2. Build the Docker image
```
docker build --build-arg AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> --build-arg AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> --build-arg AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION> . 
```
3. Run the docker image
```
docker run -d -p 8000:8000 <IMAGE_NAME>
```

## `src` is the main package folder which contains -

**Components** : Contains all components of this Project
- DataIngestion
- DataPreprocessing
- ModelTrainer
- ModelPusher

**Custom Logger and Exceptions** are used in the Project for better debugging purposes.

## Conclusion
- Regardless of the existing limitations, image captioning has already been proven to have useful applications, such as helping visually impaired people in performing daily tasks. Automatically generated descriptions can also be used for content-based retrieval or in social media communications.