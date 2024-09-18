import pandas as pd
import torchaudio.transforms
import tqdm
import os
import sys
sys.path.append('../')
import argparse
from diffusers import AudioLDM2Pipeline
# Dataset setup
musicaps = pd.read_csv(params.MUSICAPS_PATH)
tags = musicaps['ytid'].values

parser = argparse.ArgumentParser(description='TTM attribution training')
parser.add_argument('--data_split', type=int, help='data split', default=0)
parser.add_argument('--gpu', type=str, help='gpu', default='0')
args = parser.parse_args()


def main():
    # GPU Setup

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # MusicGen Model loading
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_id = "cvssp/audioldm2-music"
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    negative_prompt = "Low quality."
    generator = torch.Generator(device).manual_seed(0)


    # Saving Folder
    CURR_DATA = 'audioldm2-music'
    SAVE_PATH = os.path.join(params.DATA_PATH, CURR_DATA)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    for tag in tqdm.tqdm(tags):

        filename = os.path.join(SAVE_PATH, tag+'.wav')
        if not os.path.exists(filename):
            caption = musicaps[musicaps['ytid'] == tag]['caption'].values[0]


            audio = pipe(
                caption,
                negative_prompt=negative_prompt,
                num_inference_steps=params.num_inference_steps,
                audio_length_in_s=10.0,
                num_waveforms_per_prompt=1,
                generator=generator,
            ).audios

            sampling_rate = 16000
            # Resampling
            if sampling_rate != params.DESIRED_SR:
                audio = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate, new_freq=params.DESIRED_SR)(audio)
            audio = torch.Tensor(audio)
            torchaudio.save(filename, audio, sample_rate=params.DESIRED_SR)


if __name__ == '__main__':
    main()