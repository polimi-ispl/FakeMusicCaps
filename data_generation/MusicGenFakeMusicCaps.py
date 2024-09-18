import pandas as pd
import torchaudio.transforms
import tqdm
import os
import sys
sys.path.append('../')
import torch
import params
from transformers import AutoProcessor, MusicgenForConditionalGeneration

def main():
    # Dataset setup
    musicaps = pd.read_csv(params.MUSICAPS_PATH)
    tags_full = musicaps['ytid'].values

    # GPU Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # MusicGen Model loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium").to(device)

    # Saving Folder
    CURR_DATA = 'MusicGen_medium'
    SAVE_PATH = os.path.join(params.DATA_PATH, CURR_DATA)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    tags = []
    for tag in tqdm.tqdm(tags_full):
        filename = os.path.join(SAVE_PATH, tag+'.wav')
        if not os.path.exists(filename):
            tags.append(tag)

    for tag in tqdm.tqdm(tags):

        filename = os.path.join(SAVE_PATH, tag+'.wav')
        if not os.path.exists(filename):
            caption = musicaps[musicaps['ytid'] == tag]['caption'].values[0]

            inputs = processor(
                text=[caption],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            audio_musicgen = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=512)
            audio_musicgen = audio_musicgen.cpu().detach()[:, 0]
            sampling_rate_musicgen = model.config.audio_encoder.sampling_rate

            # Resampling
            if sampling_rate_musicgen != params.DESIRED_SR:
                audio_musicgen = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate_musicgen, new_freq=params.DESIRED_SR)(audio_musicgen)

            torchaudio.save(filename, audio_musicgen, sample_rate=params.DESIRED_SR)


if __name__ == '__main__':
    main()