#  <p align ="center" height="40px" width="40px"> Capstone Project - Derma Analysis </p>

### <p align ="center"> Implemented using: </p>
<p align ="center">
<a href="https://www.python.org/" target="_blank" rel="noreferrer">   <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/800px-Python-logo-notext.svg.png" width="32" height="32" /></a>
<a href="https://opencv.org/" target="_blank" rel="noreferrer">   <img src="https://opencv.org/wp-content/uploads/2022/05/logo.png" width="32" height="32" /></a>  
<a href="https://keras.io/" target="_blank" rel="noreferrer">   <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Keras_logo.svg/1200px-Keras_logo.svg.png" width="32" height="32" /></a> 
<a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer">   <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/115px-Tensorflow_logo.svg.png?20170429160244" width="32" height="32" /></a> 
<a href="https://scikit-learn.org/stable/" target="_blank" rel="noreferrer">   <img src="https://e7.pngegg.com/pngimages/309/384/png-clipart-scikit-learn-python-computer-icons-scikit-machine-learning-learning-text-orange.png" width="32" height="32" /></a>  
<a href="https://numpy.org/" target="_blank" rel="noreferrer">   <img src="https://numpy.org/images/logo.svg" width="32" height="32" /></a>  
<a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer">   <img src="https://seaborn.pydata.org/_images/logo-tall-lightbg.svg" width="32" height="32" /></a> 
<a href="https://matplotlib.org/" target="_blank" rel="noreferrer">   <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Created_with_Matplotlib-logo.svg/2048px-Created_with_Matplotlib-logo.svg.png" width="32" height="32" /></a>
</p>

##     <p align = "left"> Introduction </p>

This is an ongoing project for Skin Disease Analysis using machine learning.

<br>

##     <p align = "left"> Dataset </p>
Currently the repository is in training phase we are using both labeled and unlabelled data.

<br>

##     <p align = "left"> Approach </p>
 -  'part-1-pre-process': This script handles dataset loading, conducts necessary image preprocessing, and divides the data into training, validation, and test sets

 -  'part-2-data-visuals': This code is designed to display the distribution of different types of skin lesions across the training, validation, and test datasets.

 -  'part-3-extra-images': Creating more images for classes that have too few in our dataset.

 -  'model.py': The code is used to build our Xception model for analysis.

 -  'evaluate.py': script to evaluate our model for fine-tuning and deeper insights. It includes the confusion matrix, accuracy and loss histograms, and a classification report.

 -  'predict.py': Code for prediction a batch of images from a directory, using our model. 