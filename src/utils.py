import torch
import torch.nn.functional as F
from torch import pi

from src.dataset import Pose_Gaze_Intention
from src.config import (
    data_dir,
    input_n,
    output_n,
    actions,
    train_subjects,
    test_subjects,
)
from src.result_visualisation import Player_Skeleton

def load_dataset():
    train_dataset = Pose_Gaze_Intention(data_dir, train_subjects, input_n, output_n, actions, sample_rate=10)
    print("Training data size: {}".format(train_dataset.pose_gaze_intention.shape))
    test_dataset = Pose_Gaze_Intention(data_dir, test_subjects, input_n, output_n, actions)
    print("Test data size: {}".format(test_dataset.pose_gaze_intention.shape))

    return train_dataset, test_dataset

def cosine_similarity(a, b):
    # Calculate the dot product
    dot_product = torch.sum(a * b, dim=1)
    
    # Calculate the magnitude of the product of the vectors
    magnitude_product = torch.norm(a, dim=1) * torch.norm(b, dim=1)
    
    # Calculate the cosine similarity
    cosine_similarity = dot_product / magnitude_product
    
    return cosine_similarity

def evaluation(pred, base):
    list_base = []
    for list in base:
        list_base.append(torch.mean(list))
    base_mean = torch.mean(torch.tensor(list_base))

    list_pred = []
    for list in pred:
        list_pred.append(torch.mean(list))

    pred_mean = torch.mean(torch.tensor(list_pred))

    return pred_mean, base_mean

def train(num_epochs, train_loader, model, device, criterion, optimizer):
    for epoch in range(num_epochs):
        for i, inputs in enumerate(train_loader):
            
            # Move the input tensor to the GPU
            inputs = inputs.to(device)
            
            pose_gaze_data = inputs[:, :10, :66]
            seg_intention_data = inputs[:, 10, 66:]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(pose_gaze_data.float())
            loss = criterion(outputs, seg_intention_data.float())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print the loss every 100 batches
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


def test(test_loader, model, device, criterion):
    # Evaluation mode
    model.eval()

    # Lists to store the output and target
    output_list = []
    target_list = []
    theta_list = []

    # Lists to store the output and target in baseline
    target_gaze_list = []
    theta_gaze_list = []

    model = model.float()

    # Testing loop
    with torch.no_grad():
        for inputs in test_loader:
            # Move the input tensors to the GPU
            inputs = inputs.to(device)

            pose_gaze_data = inputs[:, :10, :66].float()
            seg_intention_data = inputs[:, 10, 66:].float()
            pose_data = inputs[:, 10, :63]

            # Forward pass
            outputs = model(pose_gaze_data)
            outputs_unit = F.normalize(outputs, dim=1).float()

            # Store the output and target
            output_list.append(outputs_unit)
            target_list.append(seg_intention_data)

            cos_theta = cosine_similarity(outputs_unit, seg_intention_data)
            theta = torch.acos(cos_theta)
            theta_degrees = theta * 180 / pi
            theta_list.append(theta_degrees)

            # Baseline
            seg_gaze_data = pose_gaze_data[:, -1, -3:].float() # shape (B=64, 3)
            target_gaze_list.append(seg_gaze_data)

            cos_theta_gaze_intention = cosine_similarity(seg_gaze_data, seg_intention_data)
            theta_gaze = torch.acos(cos_theta_gaze_intention) # (64)
            theta_gaze_degrees = theta_gaze * 180 / pi
            theta_gaze_list.append(theta_gaze_degrees)

    # Concatenate the output and target
    output = torch.cat(output_list)
    target = torch.cat(target_list)

    # Compute the loss
    test_loss = criterion(output, target)
    print(f'Test Loss: {test_loss:.4f}')

    # If you want to visualize the test results, uncomment the following line.
    # player = Player_Skeleton()
    # player.play_xyz(pose_data, seg_intention_data, outputs)

    return theta_list, theta_gaze_list