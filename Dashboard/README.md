# Skin Lesion Classifier from the HAM10000 dataset

This repository contains code for
1. Measuring the blurriness and contrast of images using OpenCV2 and Scikit-Image
2. Retraining an existing image classifier model using PyTorch
3. Interpreting the trained CNN models using Captum
4. Deploying the trained model as an interactive Dash server application that classifies an image the user inputs  
5. Containerization the above application as a Docker container

## Data
The HAM10000 dataset was obtained from Harvard Dataverse.
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
A 80/20 train/validation split was applied to the data on a per class basis. The final structure of data folder was
HAM10000data/train/<class>/<images> and HAM10000data/validation/<class>/<images> where the train directories contained 80% of the images per class and the validation directories contained 20% of the images per class.

The final distribution of data was:

|Class|Train|Validation
| --- | ---:| ---:|
|nv | 5365 | 1342 |
|mel | 891 | 224 |
|bkl | 880 | 221 |
|bcc | 412 | 104 |
|akiec | 263 | 66 |
|vasc | 114 | 30 |
|df | 93 | 24 |


## Retraining
We retrained 8 models on the HAM10000 data. The notebooks for trainings them are named "ImageRetraining_<model_name>.ipynb" in the repository. For example, ImageRetraining_Resnet18.ipynb contains code for retraining using the Resnet18 architecture. The overall flow of the notebooks are similar, with some important decisions highlighted below.

We augmented the data using the PyTorch dataloader by randomly flipping the images horizontally and vertically. We used weighted random sampling to draw even number of images for each class during training because the class distributions are highly unbalanced.

For Resnet18, Resnet50, ResNeXt50, MobileNetV2, and DenseNet-161, we downloaded the pretrained model from PyTorch models. For MixNet_XL and the EfficientNets, we downloaded the pretrained model using PyTorch Hub. We froze all the layers of the network for gradient descent optimization except the last classification layer. The classification layer was recoded for each models to output to the number of classes (7) in the dataset. Due to file size, only the ResNeXt model is included in the dashboard folder. This model is used for interpretability portion as well.

We retrained each of the modified models for 25 epoches. The accuracy for most models stopped increasing within the first 5 trainin epoches. We evaluated the AUC, accuracy, precision, recall, and F1 measures of each model against the validation data (using the random weighted sampling) on a class basis and on a benign/malignant basis.

The metrics for the 7 class assignment is:
|Model|AUC|Accuracy|Precision|Recall|F1  |
|-----|:---|:--------|:---------|:------|:----|
|Resnet18|0.93|0.67    |0.69     |0.67  |0.67|
|Resnet50|0.93|0.66    |0.67     |0.66  |0.66|
|Resnext50|0.92|0.67    |0.67     |0.67  |0.67|
|Densenet161|0.94|0.72    |0.73     |0.72  |0.72|
|MobileNetV2|0.92|0.65    |0.66     |0.65  |0.65|
|MixNet_Xl|0.92|0.66    |0.67     |0.66  |0.66|
|EfficientNetB0|0.92|0.65    |0.65     |0.65  |0.65|

The metrics for the 2 class malignant/benign assignment is:
|Model|AUC|Accuracy|Precision|Recall|F1  |
|-----|---|--------|---------|------|----|
|Resnet18|0.89|0.80    |0.75     |0.79  |0.77|
|Resnet50|0.89|0.80    |0.75     |0.78  |0.77|
|Resnext50|0.92|0.82    |0.79     |0.79  |0.79|
|Densenet161|0.90|0.83    |0.80     |0.82  |0.81|
|MobileNetV2|0.90|0.78    |0.74     |0.76  |0.75|
|MixNet_Xl|0.91|0.82    |0.78     |0.79  |0.79|
|EfficientNetB0|0.91|0.82    |0.79     |0.80  |0.8 |
|EfficientNetB2|0.90|0.81    |0.78     |0.77  |0.77|


## Interpretability with Captum
In the "Interpretability_Resnet50_Captum.ipynb" notebook, we apply Saliency, Integrated Gradients, Gradient Shap, Guided GradCam, and Occlusion interpretability methods using Captum to evaluate why our trained ResNeXt model evaluated one of the images as the predicted class. Interpretable machine learning is a new field, and these methods provide (different) ways to understand which portions of the image is activating (and deactivating) the classifier.

Please check the Captum library for the papers detailing how each method works as well as other interpretability methods.

## Interactive Dashboard with Plotly's Dash
We deployed the trained ResNeXt model to predicted the class of the user uploaded image as an interactive dashboard using Plotly's Dash. The user can upload an image for a skin lesion using the uploader widget, and the dashboard will classify the image as one of the 7 classes of skin lesions.

The dashboard also measures the blurriness of the image using Laplacian variance via OpenCV and the contrast of the image using Gamma via Scikit-image. If the the blurriness or contrast measure of the image is less than the preset filter (15 for Laplacian variance and 0.15 for Gamma), the dashboard will display a "Please retake your image" message. If the image passes the threshold and the predicted skin lesion type is malignant, the dashboard will display a "Please schedule patient for a consultation" message. Else, the dashboard will display a "Patient does not need another appointment" message.

The dashboard is containerized via Docker for easy deployment. The instructions for Docker is in the below section. You can also deploy the dashboard using Python via Conda or standard package managment. The Python instructions are below.

The Python and package dependencies can be cloned from the "dash_environment.yml" using the following command from the Dashboard directory assuming you have Conda installed:
```
conda env create -f dash_environment.yml
```
You can then activate the environment and deploy the code as a Flask server using gunicorn to port 8050 using the command below.
```
conda activate blur_dash
gunicorn -b 0.0.0.0:8050 dash_ham:server
```

System requirements for dashboard: The dashboard requires 4-8GB of memory using the trained ResNeXt model. More cores will speed up the computation. It is currently deployed via Elastic Container Service on a t3.large (2 vCPUs, 8 GiB) EC2 from AWS.

## Docker container
The dashboard has been containerized using the included Dockerfile in the Dashboard directory. Docker Hub has been configured to automatically build the docker image from this GitHub repository. You can check the link below for the Docker file.
https://hub.docker.com/r/jiangweiyao/skinlesionclassifierham

You can deploy the Docker container on your computer or VM using the following command using Docker
```
docker run -p 8050:8050 jiangweiyao/skinlesionclassifierham:latest
```

Alternatively, you can deploy the Docker container as an AWS web service using Elastic Container Services or Fargate. Please follow the instructions from AWS. 
