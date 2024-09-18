<div align="center">

# FakeMusicCaps: a Dataset for Detection and Attribution of Synthetic Music Generated via Text-to-Music Models 

<!-- <img width="700px" src="docs/new-generic-style-transfer-headline.svg"> -->
 
[Luca Comanducci](https://lucacoma.github.io/)<sup>1</sup>, [Paolo Bestagini](https://bestagini.faculty.polimi.it/)<sup>1</sup>, and [Stefano Tubaro](https://www.deib.polimi.it/eng/people/details/389422)<sup>1</sup>

<sup>1</sup> Dipartimento di Elettronica, Informazione e Bioingegneria - Politecnico di Milano<br>
    
[![arXiv](https://img.shields.io/badge/arXiv-2403.17864-b31b1b.svg)](https://arxiv.org/abs/2403.17864)

</div>


<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Abstract](#abstract)
- [Install & Usage](#install--usage)
- [Link to additional material](#link-to-additional-material)
- [Additional information](#additional-information)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->
    
## Abstract
Text-To-Music (TTM) models have recently revolutionized the automatic music generation research field. Specifically, by reaching superior performances to all previous state-of-the-art models and by lowering the technical proficiency needed to use them. Due to these reasons, they have readily started to be adopted for commercial uses and music production practices. This widespread diffusion of TTMs poses several concerns regarding copyright violation and rightful attribution, posing the need of serious consideration of them by the audio forensics community. In this paper, we tackle the problem of detection and attribution of TTM-generated data. We propose a dataset, FakeMusicCaps that contains several versions of the music-caption pairs dataset MusicCaps re-generated via several state-of-the-art TTM techniques. We evaluate the proposed dataset by performing initial experiments regarding the detection and attribution of TTM-generated audio.


## Install & Usage


### Intalling AudioLDM2

```
cd audio_generation
python class_generation_audioldm.py
```


### Intalling AudioGen

Please refer to the [AudioGen GitHub repo](https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md#installation) and follow the installation instructions. 

When AudioGen has been installed, you can generate the audio files running the script <i>audio_generation/class_generation_audiogen.py</i>.
Before running the script, you need to specify the path to the output folder, the audio class to generate, the prompt to use to generate the files, and the number of files to generate in the <i>audio_generation/class_generation_audiogen.py</i>. 

```
cd audio_generation
python class_generation_audiogen.py
```

### Run the code

```
pip install -r requirements.txt
```

```
python main.py
```

## Link to additional material

The full FakeMusicCaps dataset can be downloaded at [companion website](https://zenodo.org/records/13732524). 


## Additional information

For more details:
"[FakeMusicCaps: a Dataset for Detection and Attribution of Synthetic Music Generated via Text-to-Music Models](https://arxiv.org/abs/2409.10684)"


If you use code or comments from this work, please cite our paper:

```BibTex

@misc{comanducci2024fakemusiccapsdatasetdetectionattribution,
      title={FakeMusicCaps: a Dataset for Detection and Attribution of Synthetic Music Generated via Text-to-Music Models}, 
      author={Luca Comanducci and Paolo Bestagini and Stefano Tubaro},
      year={2024},
      eprint={2409.10684},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2409.10684}, 
}
```

