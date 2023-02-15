import torch.nn as nn
import torch

class BiGRU(nn.Module):
    def __init__(self, featureSize, numOutput):
        super().__init__()
        self.featureSize = featureSize
        
        # preset params
        self.featureNum = 66
        self.featureLength = int(self.featureSize/self.featureNum)

        # BiGRU Module
        self.hidden_size = 8
        self.layers = 1
        self.directions = 2 # GRU: 1
        self.GRU = nn.GRU(self.featureNum, 
                             self.hidden_size,
                             self.layers,
                             batch_first=True, 
                             bidirectional=bool(self.directions-1))

        # prdFC
        self.prdFC_inputSize = 16 # GRU: 8
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
        # x shape -> torch.Size([64, 10, 66])
        h0 = torch.zeros(self.layers*self.directions, x.size(0), self.hidden_size)      
        GruOut, _ = self.GRU(x, h0)  # torch.Size([64, 10, 8])
        GruOut = GruOut[:, -1, :] # torch.Size([64, 8])
        out = self.PrdFC(GruOut) # torch.Size([64, 3])
        return out