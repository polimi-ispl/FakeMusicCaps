import pandas as pd
import torchaudio.transforms
import tqdm
import os
import sys
sys.path.append('../')
import params
from diffusers import StableAudioPipeline

from huggingface_hub import login
login(token ='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')

def main():


    # Dataset setup
    musicaps = pd.read_csv(params.MUSICAPS_PATH)
    tags = musicaps['ytid'].values

    # GPU Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo_id = "cvssp/audioldm2"
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to(device)

    negative_prompt = "Low quality."
    generator = torch.Generator(device).manual_seed(0)


    # Saving Folder
    CURR_DATA = 'stable_audio_open'
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
                audio_end_in_s=10.0,
                num_waveforms_per_prompt=1,
                generator=generator,
            ).audios


            audio = audio.detach().cpu()
            sampling_rate = float(pipe.vae.sampling_rate) # ATTENZIONE SE COLLEGANDO TUTTO


            # Resampling
            if sampling_rate != params.DESIRED_SR:
                # Convert audio to right type
                audio = audio.to(torch.float)
                audio = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate, new_freq=params.DESIRED_SR)(audio)

            # Convert to mono
            audio = (audio[:,0] + audio[:,1])/2
            audio = torch.Tensor(audio)
            torchaudio.save(filename, audio, sample_rate=params.DESIRED_SR)


if __name__ == '__main__':
    main()