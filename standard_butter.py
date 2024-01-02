import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# function of accessing (reading) the audio file
def read_file(f_name):
    sample_rate, audio_data = wavfile.read(f_name)
    return sample_rate, audio_data

def convert_stereo_to_mono(audio_data):
    if audio_data.ndim > 1 and audio_data.shape[1] == 2:  # Check if audio is stereo
        mono_audio = np.mean(audio_data, axis=1)  # Take the average of the left and right channels
        return mono_audio.astype(audio_data.dtype)  # Ensure the dtype remains the same as the original data
    else:
        # If the audio is already mono or not in the expected shape, return the original data
        return audio_data

# plot audio file (time domain)
def plt_time_domain(sample_rate, audio_data, title):
    time = np.arange(0, len(audio_data)) / sample_rate
    plt.figure(figsize=(10, 10))
    plt.plot(time, audio_data)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid()
    plt.show()

# plot audio file (freq domain)
def plt_freq_domain(positive_freq, magnitude, title):
    plt.figure(figsize=(10, 10))
    plt.plot(positive_freq, magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.grid()
    plt.show()
    return

# function of using fourier transform
def perform_fourier_transform(audio_data, sample_rate):
    fft_result = np.fft.fft(audio_data)
    frequencies = np.fft.fftfreq(len(fft_result), 1 / sample_rate)
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    return frequencies, magnitude, phase

def Low_pass_filter(sampleRate,audioData,cut_freq,order):#eliminates high freq
    nyquist = 0.5 * sampleRate
    normal_cutf = cut_freq / nyquist
    b, a = butter(order, normal_cutf, btype='low', analog=False)
    y = filtfilt (b,a, audioData)
    return y

def High_pass_filter(sampleRate,audioData,cut_freq,order):#eliminates low freq
    nyquist = 0.5 * sampleRate
    normal_cutf = cut_freq / nyquist
    b, a = butter(order, normal_cutf, btype='high', analog=False)
    filtered_audio = filtfilt(b, a, audioData)
    return filtered_audio

def save_audio_to_file(file_path, sample_rate, audio_data):
    wavfile.write(file_path, sample_rate, np.asarray(audio_data, dtype=np.int16))

if __name__ == "__main__":
    audioFile = "names with noise .wav"   # place your audio here
    sampleRate, audioData = read_file(audioFile)

    # Convert stereo audio to mono if it's stereo
    audioData = convert_stereo_to_mono(audioData)

    plt_time_domain(sampleRate, audioData, 'Time Domain Representation Before editing')
    frequency, magnitude, phase = perform_fourier_transform(audioData, sampleRate)
    plt_freq_domain(frequency, magnitude, 'freq domain before editing')

    audioData = Low_pass_filter(sampleRate,audioData,10000,2)
    audioData= High_pass_filter(sampleRate,audioData,200, 2)
    
    plt_freq_domain(frequency, magnitude, 'freq domain after band-pass filter')
    plt_time_domain(sampleRate, audioData, 'Time Domain Representation after bandpass filter')
    # Save to a new file
    output_file_path = "butter.wav"  # Change this to your desired output filename
    save_audio_to_file(output_file_path, sampleRate, audioData)    