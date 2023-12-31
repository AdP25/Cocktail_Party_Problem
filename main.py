import numpy as np
import wave
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.io import wavfile

mix_wave_1 = wave.open("input/mix_1.wav", 'r')
mix_wave_2 = wave.open("input/mix_2.wav", 'r')
mix_wave_3 = wave.open("input/mix_3.wav", 'r')

mix_wave_1_params = mix_wave_1.getparams()
print(mix_wave_1_params)

# get length of wav file in seconds
len_of_file = 264515/44100 # nframes / framerate
print(f"length of the file in seconds : {len_of_file}")

# Converting the audio data to int16 format in the context of digital audio processing is a 
# common practice and often corresponds to the bit depth or precision of the audio samples.

# In digital audio, the bit depth refers to the number of bits used to represent the amplitude 
# of each sample. For instance, int16 refers to a signed 16-bit integer, which allows for 2^16 (65,536) discrete amplitude levels.

def extract_frames(file):
    signal_raw = file.readframes(-1)
    signal = np.frombuffer(signal_raw, dtype=np.int16)
    print("signal : ", signal)
    return signal
     

signal_1 = extract_frames(mix_wave_1)
signal_2 = extract_frames(mix_wave_2)
signal_3 = extract_frames(mix_wave_3)

# number of audio samples present in the signal
len_of_signal_1 = len(signal_1)
print(f"length : {len_of_signal_1}") 

# np.linspace() is a NumPy function that generates an array of evenly spaced numbers over a 
# specified interval. It is commonly used in numerical computations and is particularly useful for 
# creating a sequence of values, such as time values in this case.

# Syntax: np.linspace(start, stop, num)

# start: The starting value of the sequence.
# stop: The ending value of the sequence.
# num: The number of elements in the generated array.

def get_timing(file, signal):
    fs = file.getframerate()
    print("frame rate : ", fs)
    print("signal len : ", len(signal))
    print("end secs : ", len(signal)/fs)
    timing = np.linspace(0, len(signal)/fs, num=len(signal))
    return timing


timing_1 = get_timing(mix_wave_1, signal_1)
timing_2 = get_timing(mix_wave_2, signal_2)
timing_3 = get_timing(mix_wave_3, signal_3)

print("timing 1 : ", timing_1)

# plot mix wave files
def plot_wave(timing, signal, title):
    plt.figure(figsize=(12 ,2))
    plt.title(title)
    plt.plot(timing, signal)
    plt.show()

plot_wave(timing_1, signal_1, 'Recording 1')
plot_wave(timing_2, signal_2, "Recording 2")
plot_wave(timing_3, signal_3, 'Recording 3')

# zip all the signal into single list
# it combines these three lists element-wise. It creates an iterator that aggregates the 
# elements from each list into tuples. Each tuple contains one element from signal_1, 
# one from signal_2, and one from signal_3
data =  list(zip(signal_1, signal_2, signal_3))

print(data[:10])

# initialise FastICA and fit and transform data
fastica = FastICA(n_components=3)
ica_result = fastica.fit_transform(data)
print(ica_result.shape)

# split signals
result_signal_1 = ica_result[:, 0]
result_signal_2 = ica_result[:, 1]
result_signal_3 = ica_result[:, 2]

# function to plot individual components
def plot_result_signal(result_signal, title):
    plt.figure(figsize=(12, 2))
    plt.title(title)
    plt.plot(result_signal)
    plt.show()

plot_result_signal(result_signal_1, "Independent component 1")
plot_result_signal(result_signal_2, "Independent component 2")
plot_result_signal(result_signal_3, "Independent component 3")

# convert signal to int16
# Scales the result_signal to the range of a 16-bit integer. Multiplying by 32767 maps the signal to 
# the range of -32767 to 32767 (the maximum positive value for a signed 16-bit integer). 
# The additional scaling by 100 seems to further amplify the signal, possibly for normalization purposes.
def convert_to_int16(result_signal, fs, filename):
    # Normalize the signal within a suitable range (e.g., between -1 and 1)
    max_val = np.max(np.abs(result_signal))
    normalized_signal = result_signal / max_val if max_val != 0 else result_signal
    
    # Scale the normalized signal to the int16 range (-32767 to 32767)
    scaled_signal = np.int16(normalized_signal * 32767)
    
    # Write the scaled signal to a WAV audio file
    wavfile.write(filename, fs, scaled_signal)

# get framerate
fs_1 = mix_wave_1.getframerate()
fs_2 = mix_wave_2.getframerate()
fs_3 = mix_wave_3.getframerate()

# convert back to wav
convert_to_int16(result_signal_1, fs_1, "output/result_wav_1.wav")
convert_to_int16(result_signal_2, fs_2, "output/result_wav_2.wav")
convert_to_int16(result_signal_3, fs_3, "output/result_wav_3.wav")