"""
Preprocesses MusicCaps to have same code as other datasets
"""
import os
import sys
sys.path.append('../')
import torchaudio
import params_generation
import torch
from tqdm import tqdm


# Saving Folder
CURR_DATA = 'MusicCaps'
SAVE_PATH = os.path.join(params.DATA_PATH, CURR_DATA)

# Open Set
MUSIC_CAPS_PATH = os.path.join(params.PARENT_DIR,'MusicCaps') # PATH to audio file (not the same contained in params)

if not os.path.exists(os.path.join(params.DATA_PATH, 'MusicCaps')):
    os.makedirs(os.path.join(params.DATA_PATH, 'MusicCaps'))

tags = os.listdir(MUSIC_CAPS_PATH)

for tag in tqdm(tags):

    # Polish tag
    filename = os.path.join(SAVE_PATH, tag.split('-[')[0][1:-1] + '.wav')

    if not os.path.exists(filename):

        audio_caps, fs_caps = torchaudio.load(os.path.join(MUSIC_CAPS_PATH, tag))
        audio_caps_resample = torchaudio.transforms.Resample(
            orig_freq=fs_caps, new_freq=params.DESIRED_SR)(audio_caps)
        # Convert to mono
        if audio_caps_resample.shape[0] > 1:
            audio_caps_resample_mono = (audio_caps_resample[0] + audio_caps_resample[1]) / 2
            # Add ipnut channel
            audio_caps_resample_mono = audio_caps_resample_mono.unsqueeze(0)
        else:
            audio_caps_resample_mono = audio_caps_resample


        torchaudio.save(filename, audio_caps_resample_mono, sample_rate=params.DESIRED_SR)

