#!/usr/bin/env python3

"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""



"""
The model consists of four layers of LSTM and a full connection layer.
The input size of each LSTM layer is 32, and the out size of each LSTM layer is also 32.
The dropout of each LSTM layer is 0.2.
The input size of the full connection is 32 and the out size of it is 5.
Use GloVe vectors 6B, but select the 300 dim.
Divide data sets into two parts. 80% train dataset, 20% validation dataset.
The epoch is 100.
Use Adam optimizer, and set the learing rate to 0.001
"""
import torch
import torch.nn as tnn
from torchtext.vocab import GloVe

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """

    return batch

stopWords = {","}
wordVectors = GloVe(name='6B', dim=300)


###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """
    datasetLabel = datasetLabel.long()
    return datasetLabel-1

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    _, pre = torch.max(netOutput, 1)
    pre = pre.float()

    return pre + 1

###########################################################################
################### The following determines the model ####################
###########################################################################

class network(tnn.Module):
    def __init__(self,emb_dim):
        super(network,self).__init__()
        self.lstm1 = tnn.LSTM(input_size=emb_dim,hidden_size=32, dropout=0.2)
        self.lstm2 = tnn.LSTM(input_size=32,hidden_size=32, dropout=0.2)
        self.lstm3 = tnn.LSTM(input_size=32, hidden_size=32, dropout=0.2)
        self.lstm4 = tnn.LSTM(input_size=32, hidden_size=32, dropout=0.2)
        self.classify = tnn.Linear(32,5)

    def forward(self, x,length):
        x = x.permute(1,0,2)    ##[lengh, batch, emb_dim]
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        out = out[-1,:,:]
        out = self.classify(out)
        return out


net = network(300)
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""



trainValSplit = 0.8
batchSize = 32
epochs = 100
# optimiser = toptim.SGD(net.parameters(), lr=0.01)


learning_rate = 0.001
lossFunc = tnn.CrossEntropyLoss()
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
