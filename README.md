# triplet-loss-train-for-speaker-recognition
    Before, I upload a very classic VGG based model for speaker recognition . The model simply use softmax-loss to train super-parameters. But during testing stage,we found the model is not very reliable。for example, the model can easily distinguish man-man group, and man-woman group, but difficultly in woman-woman. So, we try another method called triplet-group to retrain our model, of course, we use triplet-loss as the loss for back propagation. The I upload our core code, and training curve for the two training stage. Why, I refer to "two training stage"? That need you to understand the triplet-group method. And very very welcome to my mailbox: primtee_nxg@163.com.

    The first stage, I use the classic VGG net to train our data set (just use 100 people , VOX). The net I upload before. You can find it. Of course, the net written by tensorflow. This stage,  the model converged after  250 training cycles.
    
    The second stage, upload the fist stage model as pre-training model for this stage. I just upload the 150th episode trained model as pre-training model,maybe else, the prediction acc of 150th episode model achieved 0.89.
    
    first training stage curve, first_training.png
    second training stage curve, second_training.png
    
