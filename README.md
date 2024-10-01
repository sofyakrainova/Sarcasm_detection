# Sarcasm_detection
project for NLP classification

Data for this project can be taken from here https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection  
First json files were preprocessed a bit. Imerged them in one and added the root "data", so it should look like this:  
{"data":[content of 2 json files]}  
  
======= Baseline Model ========================================  
For the baseline model I have simple 3 layers Dence model and Embedding layer. Parameters tuning gives us the next configuration:  
Embedding 12, Units1 24, Units2 16, learning_rate 0.01. After 20 epochs we have    
Train accuracy : 0.992  
Validation accurace: 0.973  
Test accuracy: 0.970
  
======== Convolutional Model ===================================  
With one 1D convolutional Layer. Parameters tuning gives us the next configuration:  
Embedding 24, Filters 32, Kernel size 4, Units 8, learning_rate 0.01. After 20 epochs of training we have:  
Train accuracy: 0.999   
Validation accurace: 0.987  
Test accuracy: 0.988  

======== Pre-defined Embeddings =================================  
For this approach we will use pre-trained embeddings from here https://nlp.stanford.edu/projects/glove/ (glove.6B.zip)
Parameters tuning gives us the next configuration:  
Units1 136, Units2 72, learning_rate 0.001.  
Pretrained embedding vector is bigger than for baseline model, so we need more epochs to train it. After 200 eposchs we have:  
Train accuracy : 0.922  
Validation accurace: 0.923  
Test accuracy: 0.926  

========= LSTM model ==========================================  
Here the simple bidirectional LSTM layer with 3 dense layers. Parameters tuning gives us the next configuration:  
Embedding 16, lstm units 20, units1 40, Units2 64, learning_rate 0.001.
After 20 epochs we have:  
Train accuracy: 0.999   
Validation accurace: 0.989  
Test accuracy: 987
