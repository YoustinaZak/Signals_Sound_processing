import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# Function to read audio file
def read_audio_file(file_path):
    sample_rate, audio_data = wavfile.read(file_path)
    return sample_rate, audio_data

def convert_stereo_to_mono(audio_data):
    if audio_data.ndim > 1 and audio_data.shape[1] == 2:  # Check if audio is stereo
        mono_audio = np.mean(audio_data, axis=1)  # Take the average of the left and right channels
        return mono_audio.astype(audio_data.dtype)  # Ensure the dtype remains the same as the original data
    else:
        # If the audio is already mono or not in the expected shape, return the original data
        return audio_data

file_path = "test.wav"
sample_rate, audio = read_audio_file(file_path)

# Convert stereo audio to mono (if it's stereo)
audio = convert_stereo_to_mono(audio)

# Plotting function for time domain representation
def plot_time_domain(sample_rate, audio_data):
    time = np.arange(0, len(audio_data)) / sample_rate
    plt.figure(figsize=(10, 6))
    plt.plot(time, audio_data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain Representation')
    plt.grid()
    plt.show()


plot_time_domain(sample_rate,audio)
# Plotting function for frequency domain representation
def plot_frequency_domain(positive_freq, magnitude):
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freq, magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Domain Representation')
    plt.grid()
    plt.show()

# Function to perform Fourier Transform
def perform_fourier_transform(audio_data, sample_rate):
    fft_result = np.fft.fft(audio_data)
    frequencies = np.fft.fftfreq(len(fft_result), 1 / sample_rate)
    positive_freq = frequencies[:len(frequencies) // 2]
    magnitude = np.abs(fft_result)[:len(frequencies) // 2]
    return positive_freq, magnitude

positive_freq, magnitude = perform_fourier_transform(audio, sample_rate)
plot_frequency_domain(positive_freq, magnitude)

# Function for low-pass filtering
def low_pass_filter(cut_freq, order, sample_rate, audio_data):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cut_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_audio = filtfilt(b, a, audio_data)
    return filtered_audio

# Function for high-pass filtering
def high_pass_filter(cut_freq, order, sample_rate, audio_data):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cut_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_audio = filtfilt(b, a, audio_data)
    return filtered_audio

# Applying high-pass filter
audio = high_pass_filter(1000, 5, sample_rate, audio)
positive_freq, magnitude = perform_fourier_transform(audio, sample_rate)
plot_frequency_domain(positive_freq, magnitude)

# Applying low-pass filter
audio = low_pass_filter(5000, 5, sample_rate, audio)
positive_freq, magnitude = perform_fourier_transform(audio, sample_rate)
plot_frequency_domain(positive_freq, magnitude)

def save_audio_to_file(file_path, sample_rate, audio_data):
    wavfile.write(file_path, sample_rate, audio_data)

def normalize_audio(audio_data):
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data_normalized = audio_data / max_val
    else:
        audio_data_normalized = audio_data
    return audio_data_normalized


def float_to_pcm16(audio_data):
    # Clip the audio data to the range of [-1, 1)
    audio_data = np.clip(audio_data, -1.0, 1.0 - np.finfo(np.float16).eps)

    # Scale the data to the range of a 16-bit integer
    pcm_data = np.int16(audio_data * 32767)
    return pcm_data


# ... (Previous code remains unchanged)

# Normalize audio data before saving
audio_normalized = normalize_audio(audio)

# Convert floating-point audio to PCM 16-bit integer format
pcm_audio = float_to_pcm16(audio_normalized)
plot_time_domain(sample_rate,pcm_audio)
# Save the modified audio to a new file
output_file_path = "Modified_Audio.wav"  # Change this to your desired output filename
save_audio_to_file(output_file_path, sample_rate, pcm_audio)