import numpy as np
import wave

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

# get timing of signal_1
def get_timing(file, signal):
    fs = file.getframerate()
    timing = np.linspace(0, len(signal)/fs, num=len(signal))
    return timing


timing_1 = get_timing(mix_wave_1, signal_1)
timing_2 = get_timing(mix_wave_2, signal_2)
timing_3 = get_timing(mix_wave_3, signal_3)

print("timing 1 : ", timing_1)