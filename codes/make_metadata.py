import os
import torch
import pickle
import numpy as np

rootDir = '/root/SpeechSplit/assets/non_native_spmel'
rootDir_f0 = '/root/SpeechSplit/assets/non_native_raptf0'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

# speaker ID 저장 
spk_id = 0;

# 데이터를 저장할 리스트
speakers = []

for speaker in sorted(subdirList):
    if speaker == '.ipynb_checkpoints':
        continue
    print('Processing speaker: %s' % speaker)


    utterances = []    
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))
    
    # use hardcoded onehot embeddings in order to be cosistent with the test speakers
    # modify as needed
    # may use generalized speaker embedding for zero-shot conversion    
    spkid = np.zeros((82,), dtype=np.float32)
    
    #spk_id에 따라 one-hot embedding
    spkid[spk_id] = 1.0
    spk_id+=1
    utterances.append(np.array([spkid]))

    files_A = sorted(os.listdir(os.path.join(rootDir,speaker)))
    files_B = sorted(os.listdir(os.path.join(rootDir_f0,speaker)))

    # 두 디렉토리의 파일을 쌍으로 묶어서 리스트에 추가
    for file_A, file_B in zip(files_A, files_B):
        if file_A == '.ipynb_checkpoints':
            continue
        if file_B == '.ipynb_checkpoints':
            continue
        A= os.path.join(rootDir,speaker,file_A)
        B= os.path.join(rootDir_f0,speaker,file_B)
        print(A)
        A_array = np.load(A)
        B_array = np.load(B)
        spect_f0 = []
        spect_f0.append(A_array)
        spect_f0.append(B_array)
        spect_f0.append(B_array.shape[0])
        utterances.append(spect_f0)

    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)

with open(os.path.join(rootDir, 'dataset_non_native.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)

    
    
#     # create file list
#     # 파일을 직접 읽어서 스펙토그램으로 데이터 저장
#     for fileName in sorted(fileList):
#         utterances.append(os.path.join(speaker,fileName))

#     # for fileName in sorted(fileList):
#     #     file_path = os.path.join(dirName, speaker, fileName)
#     #     # numpy 배열로 로드
#     #     spectrogram_data = np.load(file_path)
#     #     # 스펙트로그램 데이터를 utterances 리스트에 추가
#     #     utterances.append(spectrogram_data)
        
#     speakers.append(utterances)    
    
# with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
#     pickle.dump(speakers, handle)    