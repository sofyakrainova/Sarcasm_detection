# Sarcasm_detection
project for NLP classification

Data for this project can be taken from here https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection  
First json files were preprocessed a bit. Imerged them in one and added the root "data", so it should look like this:  
{"data":[content of 2 json files]}  
  
======= Baseline Model ========================================  
For the baseline model I have simple 3 layers Dence model and Embedding layer. After 20 epochs we have    
Train accuracy : 0.987  
Validation accurace: 0.976 
  
======== Convolutional Model ===================================  
With one 1D convolutional Layer after 20 epochs we have:  
Train accuracy: 0.9998   
Validation accurace: 0.988    

======== Pre-defined Embeddings =================================  
For this approach we will use pre-trained embeddings from here https://nlp.stanford.edu/projects/glove/ (glove.6B.zip)  
Pretrained embedding vector is bigger than for baseline model, so we need more epochs to train it. After 150 eposchs we have:  
Train accuracy : 0.951  
Validation accurace: 0.948  

========= LSTM model ==========================================  
Here the simple bidirectional LSTM layer with 3 dense layers. After 20 epochs we have:  
Train accuracy: 0.9992   
Validation accurace: 0.987  
