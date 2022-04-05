# Bone-Abnormality-Classification
Bone abnormalities classification is crucial in diagnosing Musculoskeletal Disorders (MSDs). In this research, Regnet is used with a three-layer classifier to predict the abnormality of the hand X-ray image. Data augmentation such as rotation, horizontal flip, and translation are proved to benefit the model in this task. To deal with the data imbalance, we also propose a weighted binary cross-entropy loss function. Learning rate is decay in step to achieve the local minimum of the model. Grad-Cam is used to visualize the abnormality of specific region in the image. Overall, we reach an AUC:0.82 in testing data on Kaggle after ensembling.
![network2](https://user-images.githubusercontent.com/29053630/161746574-5e4dde54-3512-4b54-95c9-52ae53a78c6f.png)
![xblock2](https://user-images.githubusercontent.com/29053630/161746579-05f57c9a-5510-43ab-bbe2-798b1f9f48f0.png)

![1](https://user-images.githubusercontent.com/29053630/161746433-96ac4dba-4a62-429e-8e20-210fdab8c9eb.png)

![XBV3q8R](https://user-images.githubusercontent.com/29053630/161746631-a69cb6c0-4ece-462e-b002-37252f830c33.png)
