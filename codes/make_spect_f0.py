import os
import sys
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
import librosa
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
from utils import butter_highpass
from utils import speaker_normalization
from utils import pySTFT
    
# mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
mel_basis = mel(sr=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)

spk2gen = pickle.load(open('/root/SpeechSplit/assets/spk2gen.pkl', "rb"))

 
# Modify as needed
rootDir = '/root/SpeechSplit/assets/non_native_wavs'
targetDir_f0 = '/root/SpeechSplit/assets/non_native_raptf0'
targetDir = '/root/SpeechSplit/assets/non_native_spmel'

   
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

for subdir in sorted(subdirList):

    # '.ipynb_checkpoints' 와 같은 숨김 디렉토리 제외
    if subdir.startswith('.'):
        continue
        
    print(subdir)
    
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    if not os.path.exists(os.path.join(targetDir_f0, subdir)):
        os.makedirs(os.path.join(targetDir_f0, subdir))    
    _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
    
    if spk2gen[subdir] == 'M':
        lo, hi = 50, 250
    elif spk2gen[subdir] == 'F':
        lo, hi = 100, 600
    else:
        raise ValueError
        
    prng = RandomState(int(subdir[1:])) 
    
    for fileName in sorted(fileList):

        #mic2제거, mic1->mic
        
        # mic1 파일 처리
        if (fileName.endswith('_mic1.flac')):
            # 새 파일 이름 생성
            new_fileName = fileName.replace('_mic1.flac', '_mic.flac')
            # 파일 전체 경로 생성
            old_file = os.path.join(dirName,subdir, fileName)
            new_file = os.path.join(dirName,subdir, new_fileName)
            # 이름 변경
            os.rename(old_file, new_file)
            print(f'Renamed: {old_file} to {new_file}')
            fileName = new_fileName
    
        # mic2 파일 제거
        elif (fileName.endswith('_mic2.flac')):
            file_to_remove = os.path.join(dirName,subdir, fileName)
            os.remove(file_to_remove)
            print(f'Removed: {file_to_remove}')
            continue

        # read audio file
        x, fs = sf.read(os.path.join(dirName,subdir,fileName))

        # 리샘플링 추가: fs가 16000이 아닌 경우 자동으로 16000으로 리샘플링
        if fs != 16000:
            print(f"Resampling {fileName} from {fs}Hz to 16000Hz")
            x = librosa.resample(x, orig_sr=fs, target_sr=16000)
            fs = 16000
            
        assert fs == 16000
        if x.shape[0] % 256 == 0:
            x = np.concatenate((x, np.array([1e-06])), axis=0)
        y = signal.filtfilt(b, a, x)
        wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
        
        # compute spectrogram
        D = pySTFT(wav).T
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = (D_db + 100) / 100        
        
        # extract f0
        f0_rapt = sptk.rapt(wav.astype(np.float32)*32768, fs, 256, min=lo, max=hi, otype=2)
        index_nonzero = (f0_rapt != -1e10)
        mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
        f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
        
        assert len(S) == len(f0_rapt)
            
        np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                S.astype(np.float32), allow_pickle=False)    
        np.save(os.path.join(targetDir_f0, subdir, fileName[:-4]),
                f0_norm.astype(np.float32), allow_pickle=False)