# Bone-Abnormality-Classification [(Homepage)](https://www.htyang.com/projects)
Bone abnormalities classification is crucial in diagnosing Musculoskeletal Disorders (MSDs). In this research, Regnet is used with a three-layer classifier to predict the abnormality of the hand X-ray image. Data augmentation such as rotation, horizontal flip, and translation are proved to benefit the model in this task. To deal with the data imbalance, we also propose a weighted binary cross-entropy loss function. Learning rate is decay in step to achieve the local minimum of the model. Grad-Cam is used to visualize the abnormality of specific region in the image. Overall, we reach an AUC:0.82 in testing data on Kaggle after ensembling.

## Download model
please download regnet.pt model from github and put in the same folder as train.py and inference.py
## Train model
train.py: <br>
#### first argument --data: input the folder of train and test <br>
## Export csv
Inference.py :  <br>
#### first argument --data: input the folder of train and test <br>
#### second argument -output <b>(only 1 dash here !!)</b> : input the output file name <br>
Example : python "/data1/home/8B07/Anthony/bone-abnormality-classification/final/inference.py" --data /data1/home/8B07/Anthony/bone-abnormality-classification/final/ -output  /data1/home/8B07/Anthony/bone-abnormality-classification/final/test.csv

## Package requirements:
requirements.txt is provided in the folder <br>

## Overall Project
[DBME5028_Midterm_Project.pdf](https://github.com/alwaysmle/Bone-Abnormality-Classification/files/8417846/DBME5028_Midterm_Project.pdf)


<p align="center">
    <img src="https://user-images.githubusercontent.com/29053630/161746574-5e4dde54-3512-4b54-95c9-52ae53a78c6f.png" width="400">
    <p align="center">General network structure in the RegNet</p>
    <br>
<p/> 
<p align="center">
    <img src="https://user-images.githubusercontent.com/29053630/161746579-05f57c9a-5510-43ab-bbe2-798b1f9f48f0.png" width="400" >
    <p align="center">The X block in the RegNet model, which consists of the residual bottleneck block and group convolution</p>
    <br>
<p/> 
<p align="center">
    <img src="https://user-images.githubusercontent.com/29053630/161747707-23188480-f5e1-4fae-ab95-a8ccf61f06c2.png" width="600" >
    <p align="center">Grad-Cam of abnormal image. Obviously, the abnormality of this image is the internal fixation device, which corresponding to the yellow spot on the heat map</p>
    <br>
<p/> 
<p align="center">
    <img src="https://user-images.githubusercontent.com/29053630/161747899-b72d0ed5-01d7-466f-9295-a12cefdeaf93.png" width="600" >
    <p align="center">Grad-Cam of fingertip amputations. The model mainly focus on the marks rather than amputation site</p>
    <br>
<p/>

