import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

#function of accessing the audio file
file = "Trial.wav"
def read_file (str):
    sampleRate , audioData= wavfile.read(str)
    return sampleRate , audioData
SR , audio = read_file(file)
print(audio)
if audio.ndim > 1:   #audio is stereo
    audio=audio[:,0]

def plt_time_domain_before(sampleRate ,audioData):
    time = np.arange(0,len(audioData))/sampleRate
    plt.figure(figsize=(10,10))
    plt.plot(time,audio)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()


if audio.ndim > 1:   #audio is stereo
    audio=audio[:,0]

def fourier_transform(audioData):
    fft_result = np.fft.fft(audioData)
    frequencies = np.fft.fftfreq(len(fft_result), 1/SR)
    positive_freq= frequencies[:len(frequencies)//2]
    magnitude = np.abs(fft_result)[:len(frequencies)//2]
    return positive_freq, magnitude

positive_freq , magnitude = fourier_transform(audio)
def plt_freq_domain(positive_freq,magnitude):
    plt.figure(figsize=(10,10))
    plt.plot(positive_freq,magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()
    
plt_freq_domain(positive_freq,magnitude)

def Low_pass_filter(cut_freq,order,sampleRate,audioData):#eliminates high freq
    nyquist = 0.5 * sampleRate
    normal_cutf = cut_freq / nyquist
    b, a = butter(order, normal_cutf, btype='low', analog=False)
    y = filtfilt (b,a, audioData)
    return y

def High_pass_filter(cut_freq,order,sampleRate,audioData):#eliminates low freq
    nyquist = 0.5 * sampleRate
    normal_cutf = cut_freq / nyquist
    b, a = butter(order, normal_cutf, btype='High', analog=False)
    y = filtfilt (b,a, audioData)
    return y
audio = High_pass_filter(2500,5,SR,audio)
pfreq, mag = fourier_transform(audio)
plt_freq_domain(pfreq,mag)

audio = Low_pass_filter(5000,5,SR,audio)
pfreq, mag = fourier_transform(audio)
plt_freq_domain(pfreq,mag)
plt_freq_domain(pfreq,mag)
#function of saving new audio
#audio_new = wave.open("","wb")
#audio_new.close()
