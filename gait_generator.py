import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFCNN(nn.Module):
    def __init__(self, input_size=5, output_size=160, hidden_size=128):
        super(SimpleFCNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Hidden to output layer
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()                          # Activation function

    def forward(self, x):
        # Apply layers with activation function
        x = self.fc1(x)         # Hidden layer
        x = nn.Dropout(0.5)(x)
        x = self.relu(x)        # Activation function
        x = self.fc2(x)         # Hidden layer
        x = nn.Dropout(0.5)(x)
        x = self.relu(x)        # Activation function
        x = self.fc3(x)         # Output layer
        return x


def denormalize(pred,normalizationconst):
    pred = pred.reshape(2,25,4)
    print(pred.shape)
    pred[0,:,0] = pred[0,:,0] * normalizationconst[0]
    pred[0,:,1] = pred[0,:,1] * normalizationconst[1]
    pred[0,:,2] = pred[0,:,2] * normalizationconst[2]
    pred[0,:,3] = pred[0,:,3] * normalizationconst[3]
    pred[1,:,0] = pred[1,:,0] * normalizationconst[4]
    pred[1,:,1] = pred[1,:,1] * normalizationconst[5]
    pred[1,:,2] = pred[1,:,2] * normalizationconst[6]
    pred[1,:,3] = pred[1,:,3] * normalizationconst[7]
    
    return pred

def pred_ifft(predictions):

    real_pred = predictions[0,:,:]
    imag_pred = predictions[0,:,:]
    predictions = real_pred + 1j*imag_pred
    
    padded_pred = np.zeros((257,4),dtype=complex)
    padded_pred[:25,:] = predictions

    padded_time = np.fft.irfft(padded_pred, axis=0)
    pred_time = padded_time[56:-56,:]

    return pred_time



def generate_gait(input_data):
    norm_consts = np.load(rf"gait reference fft_25_4.88\normalization_constants.npy")
    input_size = 3
    output_size = 200
    hidden_size = 512
    model = SimpleFCNN(input_size=input_size, output_size=output_size,hidden_size=hidden_size)
    # Load the model
    model.load_state_dict(torch.load('model_hs512_lpmse_bs64_epoch1000_fft.pth'))
    input_data[0] = input_data[0]/3 # Normalize the speed
    input_vec = torch.tensor(input_data, dtype=torch.float32)
    output_pred = model(input_vec)
    output_pred = output_pred.detach().numpy()
    output_pred = denormalize(output_pred,norm_consts)
    output_time = pred_ifft(output_pred)
    return output_time