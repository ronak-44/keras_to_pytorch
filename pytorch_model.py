import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout,Embedding,AdaptiveMaxPool1d
from torch.optim import Adam, SGD
NUM_FILTERS=32
FILTER_LENGTH1=8
FILTER_LENGTH2=12
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embedding1 = Embedding(100,128)
        self.conv1 = Conv1d(128,NUM_FILTERS, kernel_size=FILTER_LENGTH1, padding='valid',  strides=1)
        self.conv2 = Conv1d(NUM_FILTERS,NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,padding='valid',  strides=1)
        self.conv3 = Conv1d(NUM_FILTERS*2,NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,padding='valid',  strides=1)

        self.embedding2 = Embedding(100,128)(XDinput)
        self.conv4 = Conv1d(128,NUM_FILTERS, kernel_size=FILTER_LENGTH2, padding='valid',  strides=1)
        self.conv5 = Conv1d(NUM_FILTERS,NUM_FILTERS*2, kernel_size=FILTER_LENGTH2,padding='valid',  strides=1)
        self.conv6 = Conv1d(NUM_FILTERS*2,NUM_FILTERS*3, kernel_size=FILTER_LENGTH2,padding='valid',  strides=1)
        self.pooling = AdaptiveMaxPool1d(96)
        self.relu = ReLU(inplace=True)
        
        self.linear1=Linear(256*96,1024)
        self.drop = Dropout(p=0.1)
        self.linear2 =  Linear(1024,1024)
        self.linear3=Linear(1024,512)
        self.pred=Linear(512,1)



    def forward(self, XD,XT):

        XD = self.embedding1(XD)
        XD = XD.permute(0,2,1)
        XD = self.conv1(XD)
        XD = self.relu(XD)
        XD = self.conv2(XD)
        XD = self.relu(XD)
        XD = self.conv3(XD)
        XD = self.relu(XD)
        XD = self.pooling(XD)


        XT = self.embedding2(XT)
        XT = XT.permute(0,2,1)
        XT = self.conv4(XT)
        XT = self.relu(XT)
        XT = self.conv5(XT)
        XT = self.relu(XT)
        XT = self.conv6(XT)
        XT = self.relu(XT)
        XT = self.pooling(XT)

        concat = torch.cat([XD, XT],dim=1)
        concat = self.linear1(concat)
        concat = self.drop(concat)
        concat = self.linear2(concat)
        concat = self.drop(concat)
        concat = self.linear3(concat)
        concat = self.pred(concat)

        
        return concat