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
def plt_freq_domain(frequency, magnitude, title):
    plt.figure(figsize=(10, 10))
    plt.plot(frequency, magnitude)
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

def low_pass_filter_in_frequency(magnitude,freq,phase ,cuttoff_high=24000, order = 1 ):
    smpls = len(magnitude)
    mask = np.zeros(smpls)

    for index in range(0, int(0.5 *smpls) ):
        mask [index] = 1 / math.sqrt( 1+ (freq[index]/cuttoff_high )**order )
        phase[index] += - np.arctan(freq[index] / cuttoff_high)
    for index in range(int( 0.5 * smpls) ,smpls ):
        mask [ index ] = 1 / math.sqrt( 1+ (freq[index]/-cuttoff_high )**order )
        phase[index] += - np.arctan(-freq[index]/cuttoff_high)

    magnitude = magnitude * mask
    return magnitude , phase

def High_pass_filter_in_frequency(magnitude, freq ,phase,  cuttoff_low=1, order =1):
    smpls = len(magnitude)
    mask = np.zeros(smpls)
    for index in range(0 , smpls//2 ):
        mask [index] = (freq[index]/cuttoff_low) / math.sqrt( 1+ ((freq[index]/cuttoff_low )**order) )
        phase[index] += np.pi/2 - np.arctan(freq[index]/cuttoff_low)

    for index in range( smpls//2 , smpls):
        mask [index] = (freq[index]/-cuttoff_low) / math.sqrt( 1+ (freq[index]/-cuttoff_low )**order )
        phase[index] += np.pi/2 - np.arctan(-freq[index]/cuttoff_low)

    magnitude = magnitude * mask
    return magnitude, phase


def preform_inverse_fourier_transform(magnitude, phase):
    filtered = magnitude * np.exp(1j * phase)
    time_domain_signal = np.fft.ifft(filtered)
    return np.real(time_domain_signal)

def save_audio_to_file(file_path, sample_rate, audio_data):
    wavfile.write(file_path, sample_rate, np.asarray(audio_data, dtype=np.int16))
############################################################

if __name__ == "__main__":
    audioFile = "names with noise .wav"   # place your audio here
    sampleRate, audioData = read_file(audioFile)

    # Convert stereo audio to mono if it's stereo
    audioData = convert_stereo_to_mono(audioData)

    plt_time_domain(sampleRate, audioData, 'Time Domain Representation Before editing')
    frequency, magnitude, phase = perform_fourier_transform(audioData, sampleRate)
    plt_freq_domain(frequency, magnitude, 'freq domain before editing')

    magnitude, phase = low_pass_filter_in_frequency(magnitude,frequency,phase,10000,3)
    magnitude, phase= High_pass_filter_in_frequency(magnitude,frequency,phase,300,3)
    audioData =preform_inverse_fourier_transform(magnitude, phase)

    plt_freq_domain(frequency, magnitude, 'freq domain after band-pass filter')
    plt_time_domain(sampleRate, audioData, 'Time Domain Representation after bandpass filter')
    # Save to a new file
    output_file_path = "test5.wav"  # Change this to your desired output filename
    save_audio_to_file(output_file_path, sampleRate, audioData)
