from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from models.CNNBasic import CNNBasic
from models.BiGRU import BiGRU
from models.BiLSTM import BiLSTM
from models.CNN_BiGRU import CNN_BiGRU
from models.CNN_BiLSTM import CNN_BiLSTM
from src.utils import load_dataset, evaluation, train, test
from src.config import (
    feature_size,
    num_output,
    learning_rate,
    batch_size,
    num_epochs,
)

if __name__ == "__main__":

    # Load the dataset
    train_dataset, test_dataset = load_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('GPU available:', torch.cuda.is_available())

    # Init a model
    # Uncomment one of the following five models.
    model = CNNBasic(featureSize=feature_size, numOutput=num_output)
    # model = BiGRU(featureSize=feature_size, numOutput=num_output)
    # model = BiLSTM(featureSize=feature_size, numOutput=num_output)
    # model = CNN_BiGRU(featureSize=feature_size, numOutput=num_output)
    # model = CNN_BiLSTM(featureSize=feature_size, numOutput=num_output)
    
    model = model.float()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    # Train the model
    train(num_epochs, train_loader, model, device, criterion, optimizer)

    # Eval the model
    # If you want to see visualisation of test result, please uncomment the lines in the utils.py
    pred_list, baseline_list = test(test_loader, model, device, criterion)
    error_model, error_baseline = evaluation(pred_list, baseline_list)
    print(f'Your model predition error angle is {error_model:.2f}.\nThe baseline error angle is {error_baseline:.2f}')
    