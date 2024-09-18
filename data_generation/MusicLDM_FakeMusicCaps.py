import pandas as pd
import torchaudio.transforms
import tqdm
import os
import sys
sys.path.append('../')
import params
from diffusers import MusicLDMPipeline


def main():

    # Dataset setup
    musicaps = pd.read_csv(params.MUSICAPS_PATH)
    tags = musicaps['ytid'].values

    # GPU Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    import torch
    # MusicGen Model loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_id = "ucsd-reach/musicldm"
    pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)


    # Saving Folder
    CURR_DATA = 'musicldm'
    SAVE_PATH = os.path.join(params.DATA_PATH, CURR_DATA)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    for tag in tqdm.tqdm(tags):

        filename = os.path.join(SAVE_PATH, tag+'.wav')
        if not os.path.exists(filename):
            caption = musicaps[musicaps['ytid'] == tag]['caption'].values[0]

            audio = pipe(caption, num_inference_steps=params.num_inference_steps, audio_length_in_s=10.0).audios[0]
            sampling_rate = 16000
            # Resampling
            if sampling_rate != params.DESIRED_SR:
                audio = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate, new_freq=params.DESIRED_SR)(audio)
            audio = torch.Tensor(audio).unsqueeze(0) # add channel
            torchaudio.save(filename, audio, sample_rate=params.DESIRED_SR)


if __name__ == '__main__':
    main()