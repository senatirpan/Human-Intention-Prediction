import torch.nn as nn
import torch
from math import floor

class CNN_BiGRU(nn.Module):
    def __init__(self, featureSize, numOutput):
        super().__init__()
        self.featureSize = featureSize
        
        # preset params
        self.featureNum = 66
        self.featureLength = int(self.featureSize/self.featureNum)
        
        # CNN1D Module
        CNN1D_outChannels1 = 16
        CNN1D_kernelSize1 = 1
        CNN1D_featureSize1 = floor((self.featureLength - CNN1D_kernelSize1 + 1)/2)
        CNN1D_outChannels2 = 16
        CNN1D_kernelSize2 = 1
        CNN1D_featureSize2 = floor((CNN1D_featureSize1 - CNN1D_kernelSize2 + 1)/2)
        CNN1D_outChannels3 = 16
        CNN1D_kernelSize3 = 1
        CNN1D_featureSize3 = floor((CNN1D_featureSize2 - CNN1D_kernelSize3 + 1)/2)
        self.CNN1D_outputSize = CNN1D_featureSize3 * CNN1D_outChannels3
        self.CNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.featureNum, out_channels=CNN1D_outChannels1,kernel_size=CNN1D_kernelSize1),
            nn.BatchNorm1d(CNN1D_outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=CNN1D_outChannels1, out_channels=CNN1D_outChannels2,kernel_size=CNN1D_kernelSize2),
            nn.BatchNorm1d(CNN1D_outChannels2),
            nn.ReLU(),
            #nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=CNN1D_outChannels2, out_channels=CNN1D_outChannels3,kernel_size=CNN1D_kernelSize3),
            nn.BatchNorm1d(CNN1D_outChannels3),
            nn.ReLU(),
            #nn.MaxPool1d(2),    
             )
        
        # BiGRU Module
        self.GRU_hidden_size = 8
        self.GRU_layers = 1
        self.GRU_directions = 2
        self.GRU = nn.GRU(CNN1D_outChannels3,self.GRU_hidden_size, self.GRU_layers, batch_first=True, bidirectional=bool(self.GRU_directions-1))
        
        # task prediction FC Module
        prdFC_inputSize = 16
        prdFC_linearSize1 = 64
        prdFC_linearSize2 = 64
        prdFC_dropoutRate = 0.5
        prdFC_outputSize = numOutput
        # the prediction fc layers
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.BatchNorm1d(prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize2, prdFC_outputSize)
             )
        
    def forward(self, x):
        feature = x.permute(0,2,1) # torch.Size([64, 66, 10])       
        featureOut = self.CNN1D(feature) # torch.Size([64, 16, 5])
        featureOut = featureOut.permute(0,2,1) # torch.Size([64, 5, 16])  
        
        h0 = torch.zeros(self.GRU_layers*self.GRU_directions, x.size(0), self.GRU_hidden_size)
        GruOut, _ = self.GRU(featureOut, h0) # torch.Size([64, 5, 8])
        GruOut = GruOut[:, -1, :] # torch.Size([64, 8])
        
        out = self.PrdFC(GruOut) # torch.Size([64, 3])
        return out