"""
Preprocesses SunoCaps to have same code as other datasets
"""
import os
import sys
sys.path.append('../')
import torchaudio
import torch
import utils
import params
from tqdm import tqdm


# Saving Folder
CURR_DATA = 'SunoCaps'
SAVE_PATH = os.path.join(params.DATA_PATH, CURR_DATA)
# Open Set

if not os.path.exists(os.path.join(params.DATA_PATH, 'SunoCaps')):
    os.makedirs(os.path.join(params.DATA_PATH, 'SunoCaps'))

tags_suno = os.listdir(params.SUNOCAPS_PATH)

for tag in tqdm(tags_suno):

    # Let's keep only label 4 (why? because this is caption-based, as written in [1])
    if tag.split('_')[1].split('.')[0]:
        filename = os.path.join(SAVE_PATH, tag.split('.')[0].split('_')[0] + '.wav')

        audio_suno, fs_suno = torchaudio.load(os.path.join(params.SUNOCAPS_PATH, tag))
        audio_suno_resample = torchaudio.transforms.Resample(
            orig_freq=fs_suno, new_freq=params.DESIRED_SR)(audio_suno)

        #Convert to mono
        audio_suno_resample_mono = (audio_suno_resample[0] + audio_suno_resample[1]) / 2
        audio_suno_resample_mono = utils.normalize_tensor(audio_suno_resample_mono)

        audio_suno_resample_mono = torch.Tensor(audio_suno_resample_mono).unsqueeze(0)
        torchaudio.save(filename, audio_suno_resample_mono, sample_rate=params.DESIRED_SR)

