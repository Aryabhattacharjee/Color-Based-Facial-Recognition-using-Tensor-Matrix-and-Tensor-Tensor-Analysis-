# Color-Based-Facial-Recognition-using-Tensor-Matrix-and-Tensor-Tensor-Analysis

## Introduction
This algorithm is designed to extract color information from images while simultaneously reducing dimensionality, without altering or discarding any part of the original data structure. The key features of Algorithm 3 include:  
  
Tensor-Tensor Multiplication and Decomposition: This method uses tensor-tensor multiplication and decomposition techniques, which are more efficient and preserve the tensor structure of the data better than traditional matrix-based approaches.  
  
Eigen-Tensor Concept: The introduction of the Eigen-Tensor allows for a robust framework in face recognition tasks, as it leverages the full multi-dimensional nature of the data, ensuring that essential features are extracted without losing critical information.  
  
Dimensionality Reduction: The algorithm effectively reduces the dimensionality of the data, making the recognition process more computationally efficient while maintaining high accuracy.  


## Dataset
Download the dataset from this link:- https://fei.edu.br/~cet/facedatabase.html  
Extract the images in a folder.  
![image](https://github.com/user-attachments/assets/750013e9-43f1-4f43-99a9-76dad71a2c5b)


## Algorithm
### Input:  
The input consists of face images organized in a way that each face of person i is represented as a 3D tensor Person(p,n), where: p denotes the person index (1 to P). n denotes the image index of that person (1 to N). Each image is a 3D tensor rather than a 2D matrix, capturing the color information across different channels.  
### Training Stage:  
#### Initialize Mean Tensor:  
Initialize a tensor meanAll to store the average tensor of all images across all persons.  
  
#### Calculate Mean Tensor (meanAll):  

Iterate over all persons p and all their images n.  
Accumulate the tensors for each person and each image into meanAll.  
After summing, divide meanAll by the total number of images (NP, where P is the number of persons, and N is the number of images per person) to get the mean tensor.  
#### Compute Covariance Tensor (TensorCov):  

For each person p and each image n:  
Subtract meanAll from each tensor Person(p,n), effectively centering the data.  
Perform tensor-tensor multiplication between the centered tensor and its transpose, adding the result to TensorCov.  
Normalize TensorCov by dividing it by NP.  
#### Tensor Singular Value Decomposition (SVD):  

Perform Tensor SVD on TensorCov, resulting in three components: TenU, TenS, and TenV.  
TenU, TenS, and TenV are the tensor equivalents of the matrices obtained from the traditional SVD.  
#### Feature Extraction:  

For each person p and image n, compute the tensor features TenFeature(p,n) by multiplying the tensor Person(p,n) with the leading lateral slices of TenU, determined by the number of the most valuable coefficients in TenS.  
The output is a lower-dimensional tensor feature representation of the original image.  
### Evaluation Stage:  
#### Transform Target Image:  

For a target image Jm (where m ranges from 1 to M):  
Compute its tensor feature TenFeature(j) using the same projection onto TenU as done during the training stage.  
#### Recognition:  

Compare the tensor feature of the target image TenFeature(j) with the tensor features of all the training images TenFeature(p,n) by calculating some form of distance or similarity (this step is indicated by Eval).  
Identify the person p with the minimum distance or maximum similarity as the recognized person.  
