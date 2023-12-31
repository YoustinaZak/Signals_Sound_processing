import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft, fftfreq, ifft

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
    plt.figure(figsize=(10, 6))
    plt.plot(time, audio_data)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid()
    plt.show()


# plot audio file (freq domain)
def plt_freq_domain(positive_freq, magnitude, title):
    plt.figure(figsize=(10, 6))
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
    positive_freq = frequencies[:len(frequencies) // 2]
    magnitude = np.abs(fft_result)[:len(frequencies) // 2]
    phase = np.angle(fft_result)[:len(frequencies) // 2]
    return positive_freq, magnitude, phase


# apply filter lowpass and high pass filter
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


def save_audio_to_file(file_path, sample_rate, audio_data):
    wavfile.write(file_path, sample_rate, np.asarray(audio_data, dtype=np.int16))


def normalize_audio(audio_data):
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data_normalized = audio_data / max_val
    else:
        audio_data_normalized = audio_data
    return audio_data_normalized

def band_filter_in_frequency (magnitude , cuttoff_low,cuttoff_high ):
    magnitude= magnitude [:len(magnitude)//2]
    smpls = len(magnitude)
    mask = np.zeros(smpls)
    mask[int((cuttoff_low/24000)* smpls): int((cuttoff_high/24000)* smpls)] = 1
    magnitude=magnitude*mask
    magnitude = np.concatenate((magnitude, magnitude[::-1]))
    return magnitude
    
def preform_inverse_fourier_transform (magnitude, phase):
    filtered = magnitude * np.exp(1j * phase)
    time_domain_signal = np.fft.ifft(filtered)
    return time_domain_signal
############################################################

if __name__ == "__main__":
    audioFile = "tst.wav"
    sampleRate, audioData = read_file(audioFile)
    print(audioData[1000:2000])
    # Convert stereo audio to mono if it's stereo
    audioData = convert_stereo_to_mono(audioData)

    plt_time_domain(sampleRate, audioData, 'Time Domain Representation Before editing')
    positiveFreq, magnitude, phase = perform_fourier_transform(audioData, sampleRate)
    plt_freq_domain(positiveFreq, magnitude, 'freq domain before editing')

    magnitude = band_filter_in_frequency(magnitude,8000,20000)
    # # Applying high-pass filter
    # audioData = high_pass_filter(6000, 5, sampleRate, audioData)
    # # Applying low-pass filter
    # audioData = low_pass_filter(8000, 5, sampleRate, audioData)

    #positiveFreq, magnitude = perform_fourier_transform(audioData, sampleRate)
    plt_freq_domain(positiveFreq, magnitude, 'freq domain after band-pass filter')
    #audioData = preform_inverse_fourier_transform(magnitude,phase)
    audioData = ifft(magnitude).real
    plt_time_domain(sampleRate, audioData, 'Time Domain Representation after band-pass filter')
    print(audioData[1000:2000])
    #audioNormalized = normalize_audio(audioData)

    # Save to a new file
    output_file_path = "Modified_Audio.wav"  # Change this to your desired output filename
    save_audio_to_file(output_file_path, sampleRate,audioData)
