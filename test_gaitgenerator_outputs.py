import torch
from gait_generator_net import SimpleFCNN
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample


def denormalize(pred,normalizationconst):
    #form is [5,2,17]
    for i in range(17):
        for k in range(2):
            pred[:,k,i] = pred[:,k,i] * normalizationconst[i*2+k]
    return pred


def pred_ifft(predictions,dt):
    #form is [5,2,17]
    real_pred = predictions[:,0,:]
    imag_pred = predictions[:,1,:]
    predictions = real_pred + 1j*imag_pred

    pred_time = np.fft.irfft(predictions, axis=1)
    pred_time = pred_time.transpose(1,0)
    org_rate = 10

    if dt < 0.1:
        num_samples = int((pred_time.shape[0]) * (1/dt)/(org_rate))  # resample with self.dt
        # Upsample using Fourier method
        pred_time_resampled = resample(pred_time, num_samples, axis=0)
    return pred_time, pred_time_resampled


model = SimpleFCNN(input_size=3, output_size=204, hidden_size=512)
normalizationconst = np.load(r'C:\Users\yusuf\Bipedal-imitation-rl\newnormalization_constants.npy')
model.load_state_dict(torch.load(r'C:\Users\yusuf\Bipedal-imitation-rl\final_model.pth'))
model.eval()  # Set model to evaluation mode to disable dropout
input_vec = torch.ones(1,3)
input_vec[0] = 0.07
input_vec[1:] = input_vec[1:] * 0.6
with torch.no_grad():  # Disable gradient computation for inference
    freqs = model(input_vec)
predictions = freqs.reshape(-1,6,2,17)
predictions = predictions.detach().numpy()
predictions = predictions[0]
predictions = denormalize(predictions,normalizationconst)
pred_time, pred_time_resampled = pred_ifft(predictions,0.01)
print(predictions)


fig, axs = plt.subplots(2, 2, figsize=(10, 6))
# Plot columns 0 and 1 in row 0
axs[0, 0].grid(True)
axs[0, 1].grid(True)
axs[0, 0].plot(pred_time[:, 0])
axs[0, 0].set_title("pred_time[:, 0]")
axs[0, 1].plot(pred_time[:, 1])
axs[0, 1].set_title("pred_time[:, 1]")

# Plot columns 3 and 4 in row 1
axs[1, 0].grid(True)
axs[1, 1].grid(True)
axs[1, 0].plot(pred_time[:, 3])
axs[1, 0].set_title("pred_time[:, 3]")
axs[1, 1].plot(pred_time[:, 4])
axs[1, 1].set_title("pred_time[:, 4]")

fig2, axs2 = plt.subplots(2, 2, figsize=(10, 6))
# Plot columns 0 and 1 in row 0
axs2[0, 0].grid(True)
axs2[0, 1].grid(True)
axs2[0, 0].plot(pred_time_resampled[:, 0])
axs2[0, 0].set_title("pred_time_resampled[:, 0]")
axs2[0, 1].plot(pred_time_resampled[:, 1])
axs2[0, 1].set_title("pred_time_resampled[:, 1]")

# Plot columns 3 and 4 in row 1
axs2[1, 0].grid(True)
axs2[1, 1].grid(True)
axs2[1, 0].plot(pred_time_resampled[:, 3])
axs2[1, 0].set_title("pred_time_resampled[:, 3]")
axs2[1, 1].plot(pred_time_resampled[:, 4])
axs2[1, 1].set_title("pred_time_resampled[:, 4]")

plt.tight_layout()
plt.show()




