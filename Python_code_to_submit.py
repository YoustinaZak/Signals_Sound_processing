import numpy as np
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
    positive_freq = frequencies[:len(frequencies) // 2]
    magnitude = np.abs(fft_result)[:len(frequencies) // 2]
    return positive_freq, magnitude


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


def float_to_pcm16(audio_data):
    # Clip the audio data to the range of [-1, 1)
    audio_data = np.clip(audio_data, -1.0, 1.0 - np.finfo(np.float16).eps)

    # Scale the data to the range of a 16-bit integer
    pcm_data = np.int16(audio_data * 32767)
    return pcm_data


############################################################

if __name__ == "__main__":
    audioFile = "trial.wav"
    sampleRate, audioData = read_file(audioFile)

    # Convert stereo audio to mono if it's stereo
    audioData = convert_stereo_to_mono(audioData)

    plt_time_domain(sampleRate, audioData, 'Time Domain Representation Before editing')
    positiveFreq, magnitude = perform_fourier_transform(audioData, sampleRate)
    plt_freq_domain(positiveFreq, magnitude, 'freq domain before editing')

    # Applying high-pass filter
    audioData = high_pass_filter(200, 5, sampleRate, audioData)
    # Applying low-pass filter
    audioData = low_pass_filter(8000, 5, sampleRate, audioData)

    positiveFreq, magnitude = perform_fourier_transform(audioData, sampleRate)
    plt_freq_domain(positiveFreq, magnitude, 'freq domain after band-pass filter')

    audioNormalized = normalize_audio(audioData)

    # removing floating points
    pcmAudio = float_to_pcm16(audioNormalized)
    plt_time_domain(sampleRate, pcmAudio, 'Time Domain Representation after editing')

    # Save to a new file
    output_file_path = "Modified_Audio.wav"  # Change this to your desired output filename
    save_audio_to_file(output_file_path, sampleRate, pcmAudio)
