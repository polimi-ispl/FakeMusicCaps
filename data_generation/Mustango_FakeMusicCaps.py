import os
import sys
sys.path.append('../')
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.cuda.set_per_process_memory_fraction(0.9, device=0)
import pandas as pd
import torchaudio.transforms
import tqdm
import params_generation
from mustango import Mustango

def main():

    # Dataset setup
    musicaps = pd.read_csv(params.MUSICAPS_PATH)
    tags_full = musicaps['ytid'].values

    # GPU Setup
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # MusicGen Model loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Mustango("declare-lab/mustango", device=device)
    #model = model.to(device)

    # Saving Folder
    CURR_DATA = 'mustango'
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
            try:
                caption = musicaps[musicaps['ytid'] == tag]['caption'].values[0]
                #audio = pipe(caption, num_inference_steps=params.num_inference_steps, audio_length_in_s=10.0).audios[0]

                audio = model.generate(caption)
                audio = audio / 32768 # Mustango generates 16 bit audio


                #print(audio.shape)
                sampling_rate =16000 # ATTENZIONE SE COLLEGANDO TUTTO


                # Resampling
                if sampling_rate != params.DESIRED_SR:
                    # Convert audio to right type
                    audio = audio.to(torch.float)
                    audio = torchaudio.transforms.Resample( orig_freq=sampling_rate, new_freq=params.DESIRED_SR)(audio)
                # Convert to mono
                audio = torch.Tensor(audio).unsqueeze(0)
                torchaudio.save(filename, audio, sample_rate=params.DESIRED_SR)
            except:
                print(caption)


if __name__ == '__main__':
    main()