"""
Contains the data
"""

import torch
import torchaudio
import os
import utils
# https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html

SUNOCAPS_PATH = '/nas/home/lcomanducci/MIR/dfm/dataset/FakeMusicCaps/SunoCaps'
DATASET_PATH ='/nas/home/lcomanducci/MIR/dfm/dataset/FakeMusicCaps'

# Create classes dictionary
models_names = os.listdir(DATASET_PATH)

# TEMP TEMP TEMP TEMP TEMP TEMP
models_names = ['MusicCaps', 'MusicGen_medium', 'musicldm', 'audioldm2', 'stable_audio_open', 'mustango']
model_labels = {}
class_idx = 0
for name in models_names:
    model_labels.update({name: class_idx})
    class_idx += 1

test_suno = os.listdir(SUNOCAPS_PATH)
test_files = []
data_files = []
for name in models_names:
    for track in os.listdir(os.path.join(DATASET_PATH, name)):
        if track in test_suno:
            test_files.append(os.path.join(DATASET_PATH, name, track))
        else:
            data_files.append(os.path.join(DATASET_PATH, name, track))


class MusicDeepFakeDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, class_dictionary,AUDIO_LENGTH_SECONDS, FS=16000, feat_type='raw'):
        self.data_paths = data_paths
        self.class_dictionary =class_dictionary
        self.AUDIO_LENGTH_SAMPLES = AUDIO_LENGTH_SECONDS*FS
        self.feat_type = feat_type

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):

        data_path = self.data_paths[idx]
        audio, _ = torchaudio.load(data_path)
        #print(data_path)
        # Make sure audio tracks have desired length
        if  audio.shape[1] > self.AUDIO_LENGTH_SAMPLES:
            idx_slice = torch.randint(low=0, high=(audio.shape[1] - int(self.AUDIO_LENGTH_SAMPLES)), size=())
            audio = audio[:, idx_slice:idx_slice + int(self.AUDIO_LENGTH_SAMPLES)]

        elif audio.shape[1] < self.AUDIO_LENGTH_SAMPLES:
            padding_length = self.AUDIO_LENGTH_SAMPLES - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, int(padding_length)), 'constant', 0)
        audio = audio.squeeze(0)

        # Normalize audio
        if torch.sum(audio) > 0:
            audio = utils.normalize_tensor(audio)
        label = self.class_dictionary[data_path.split('/')[-2]]
        label = torch.Tensor([label])

        audio = audio.unsqueeze(0)

        if self.feat_type == 'freq':
            logSTFT = torch.log(torch.abs(torch.stft(audio, n_fft=512, hop_length=128,
                                     window=torch.hann_window(window_length=512),
                                     center=True, return_complex=True))+1e-3)
            if torch.sum(audio) > 0:
                logSTFT = utils.normalize_tensor(logSTFT)

            return logSTFT, label
        else:
            return audio, label


train_set, val_set = utils.split_list(data_files, ratio=0.8)




