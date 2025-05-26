import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
y, sr = librosa.load('blues.00000.wav', sr=None)  # Replace with your .wav file path

# Generate mel-spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=150)

# Convert to log scale (dB)
S_dB = librosa.power_to_db(S, ref=np.max)

# Plot and save the mel-spectrogram
plt.figure(figsize=(1.5, 1.5), dpi=100)  # 1.5 inches * 100 dpi = 150 pixels
librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis=None, y_axis=None, cmap='gray')
plt.axis('off')  # Remove axes
plt.tight_layout(pad=0)
plt.savefig('mel_spectrogram.png', bbox_inches='tight', pad_inches=0)
plt.close()
