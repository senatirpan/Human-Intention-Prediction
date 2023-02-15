import torch.nn as nn
from math import floor

class CNNBasic(nn.Module):
    def __init__(self, featureSize, numOutput):
        super(CNNBasic, self).__init__()
        self.featureSize = featureSize
        self.featureNum = 66
        self.featureLength = int(self.featureSize/self.featureNum)
        
        outChannels1 = 16
        KernelSize1 = 1
        featureSize1 = floor((self.featureLength - KernelSize1 + 1)/2)
        outChannels2 = 16
        KernelSize2 = 1
        featureSize2 = floor((featureSize1 - KernelSize2 + 1)/2)
        outChannels3 = 16
        KernelSize3 = 1
        featureSize3 = floor((featureSize2 - KernelSize3 + 1)/2)
        self.outputSize = featureSize3 * outChannels3

        self.CNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.featureNum,
                      out_channels=outChannels1, kernel_size=KernelSize1),
            nn.BatchNorm1d(outChannels1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=outChannels1, out_channels=outChannels2, kernel_size=KernelSize2),
            nn.BatchNorm1d(outChannels2),
            nn.ReLU(),
            # nn.MaxPool1d(2),            
            nn.Conv1d(in_channels=outChannels2, out_channels=outChannels3, kernel_size=KernelSize3),
            nn.BatchNorm1d(outChannels3),
            nn.ReLU(),
            # nn.MaxPool1d(2),    
            )

        # prdFC
        self.prdFC_inputSize = 80
        self.prdFC_linearSize1 = 64
        self.prdFC_linearSize2 = 64
        self.prdFC_dropoutRate = 0.5
        self.prdFC_outputSize = numOutput
        self.PrdFC = nn.Sequential(
            nn.Linear(self.prdFC_inputSize, self.prdFC_linearSize1),
            nn.BatchNorm1d(self.prdFC_linearSize1),
            nn.ReLU(),
            nn.Dropout(p = self.prdFC_dropoutRate),
            nn.Linear(self.prdFC_linearSize1, self.prdFC_linearSize2),
            nn.BatchNorm1d(self.prdFC_linearSize2),
            nn.ReLU(),
            nn.Dropout(p = self.prdFC_dropoutRate),
            nn.Linear(self.prdFC_linearSize2, self.prdFC_outputSize)
            )

    def forward(self, x):
        feature = x.permute(0,2,1) # torch.Size([64, 66, 10])     
        featureOut = self.CNN1D(feature) # torch.Size([64, 16, 5])
        featureOut = featureOut.permute(0,2,1) # torch.Size([64, 5, 16]) 
        out = featureOut.reshape(-1, self.prdFC_inputSize) # torch.Size([64, 80])
        out = self.PrdFC(out) # torch.Size([64, 3])
        return out