import numpy as np
import librosa
from scipy import signal
from scipy.io import wavfile
from scipy.signal import butter,filtfilt
from scipy.signal import cwt
from scipy.signal import hilbert
from scipy.signal import resample
from scipy.signal import decimate
from scipy.signal import spectrogram
from scipy.signal.windows import get_window
import av
import numpy as np

import os

from .feature_class import features 


def classify_cough(x, fs, model):
    """Classify whether an inputted signal is a cough or not using filtering, feature extraction, and ML classification
    Inputs: 
        x: (float array) raw cough signal
        fs: (int) sampling rate of raw signal
        model: cough classification ML model loaded from file
    Outputs:
        result: (float) probability that a given file is a cough 
    """
    label_predict = {0:'Healthy Cough',1:'Unhealthy Cough'} 
    x,fs = preprocess_cough(x,fs)
    data = (fs,x)
    FREQ_CUTS = [(0,200),(300,425),(500,650),(950,1150),(1400,1800),(2300,2400),(2850,2950),(3800,3900)]
    features_fct_list = ['spectral_features']
    feature_values_vec = []
    obj = features(FREQ_CUTS)
    for feature in features_fct_list:
        feature_values, _ = getattr(obj,feature)(data)
        feature_values_vec.append(feature_values[5])
    feature_values_vec = np.array(feature_values_vec)
    # feature_values_vec = feature_values_vec.reshape(feature_values_vec[0],1)
    
    result = model.predict(feature_values_vec)
    prediction = label_predict[result[0]] 
    
    return prediction

def preprocess_cough(x,fs, cutoff = 6000, normalize = True, filter_ = True, downsample = True):
    """
    Normalize, lowpass filter, and downsample cough samples in a given data folder 
    
    Inputs: x*: (float array) time series cough signal
    fs*: (int) sampling frequency of the cough signal in Hz
    cutoff: (int) cutoff frequency of lowpass filter
    normalize: (bool) normailzation on or off
    filter: (bool) filtering on or off
    downsample: (bool) downsampling on or off
    *: mandatory input
    
    Outputs: x: (float32 array) new preprocessed cough signal
    fs: (int) new sampling frequency
    """
    
    fs_downsample = cutoff*2
    
    #Preprocess Data
    if len(x.shape)>1:
        x = np.mean(x,axis=1)                          # Convert to mono
    if normalize:
        x = x/(np.max(np.abs(x))+1e-17)                # Norm to range between -1 to 1
    if filter_:
        b, a = butter(4, fs_downsample/fs, btype='lowpass') # 4th order butter lowpass filter
        x = filtfilt(b, a, x)
    if downsample:
        x = signal.decimate(x, int(fs/fs_downsample)) # Downsample for anti-aliasing
    
    fs_new = fs_downsample

    return np.float32(x), fs_new

def to_wav(in_path: str, out_path: str = None, sample_rate: int = 16000) -> str:
    if out_path is None:
        out_path = os.path.splitext(in_path)[0] + '.wav'
    with av.open(in_path) as in_container:
        in_stream = in_container.streams.audio[0]
        with av.open(out_path, 'w', 'wav') as out_container:
            out_stream = out_container.add_stream(
                'pcm_s16le',
                rate=sample_rate,
                layout='mono'
            )
            for frame in in_container.decode(in_stream):
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)

    return out_path