# triplet-loss-train-for-speaker-recognition
    Before, I upload a very classic VGG based model for speaker recognition . 
    The model simply use softmax-loss to train super-parameters. 
    But during testing stage,we found the model is not very reliable.
    For example, the model can easily distinguish man-man group, and man-woman group, 
    but difficultly in woman-woman. 
    So, we try another method called triplet-group to retrain our model.
    Of course, we use triplet-loss as the loss for back propagation. 
    Then I upload our core code, and training curve for the two training stage. 
    Why, I refer to "two training stage"? That need you to understand the triplet-group method. 
    And very very welcome to my mailbox: primtee_nxg@163.com.

    The first stage(See train.py), I use the classic VGG net to train our data set (just use 100 people , VOX). 
    The net I upload before. You can find it. Of course, the net written by tensorflow. 
    This stage,  the model converged after  250 training cycles. 
    
    The second stage(See triplst_train.py), upload the fist stage model as pre-training model for this stage. 
    I just upload the 150th episode trained model as pre-training model,maybe else.
    The prediction acc of 150th episode model achieved 0.89.
    
    first training stage curve, first_stage_training.png
    second training stage curve, second_stage_training.png
    
    Test model
      request :curl "http://192.168.99.214:19001/recognize?filename=/data/register/188188.wav&ranges=188881|190001"
      cosin score are: [[0.9889], [0.3467]]
      results : 188881
    We may enrollment more than 100 people for one time, but just three of them needed in that snene. So, the ranges param means the exact people we need.
    http://192.168.99.214:19001: is the voiceprint server IP.
    recognize: means the type of voiceprint
    filename: means the voice path
    
    
