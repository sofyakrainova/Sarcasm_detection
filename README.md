# Sarcasm_detection
project for NLP classification

Data for this project can be taken from here https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection  
First json files were preprocessed a bit. Imerged them in one and added the root "data", so it should look like this:  
{"data":[content of 2 json files]}  
  
======= Baseline Model =====================================  
For the baseline model I have simple 3 layers Dence model and Embedding layer. After 20 epochs we have    
Train accuracy : 0.99  
Validation accurace: 0.98  
  
======== Convolutional Model ===================================
