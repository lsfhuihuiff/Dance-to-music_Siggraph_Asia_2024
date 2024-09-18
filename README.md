# Dance-to-music_Siggraph_Aisa_2024
The official code for [“Dance-to-Music Generation with Encoder-based Textual Inversion“](https://arxiv.org/abs/2401.17800)
![Teaser](./teaser.png)
Demo are available at [Dance-to-music](https://youtu.be/y2pG2S5xDLY).

More results and comparisons with other methods are available at [Dance-to-music](https://lsfhuihuiff.github.io/dance2music.github.io/). 
## Installation
It requires Python 3.9, PyTorch 2.0.0. You can run the following:

```shell
# Best to make sure you have torch installed first, in particular before installing xformers.
# Don't run this if you already have PyTorch installed.
pip install 'torch>=2.0'
pip install -e .  # or if you cloned the repo locally (mandatory if you want to train).
```

We also recommend having `ffmpeg` installed, either through your system or Anaconda:
```bash
sudo apt-get install ffmpeg
# Or if you are using Anaconda or Miniconda
conda install "ffmpeg<5" -c conda-forge
```

## Models
We use the pre-trained facebook/musicgen-small model of MusicGen.
* [MusicGen](./docs/MUSICGEN.md): A state-of-the-art controllable text-to-music model.

## Train the model

```bash
dora run solver=musicgen/musicgen_base_32khz model/lm/model_scale=medium continue_from=/path/to/pretrained/model conditioner=text2music
```
More details about training, please refer to
the [training documentation](./docs/TRAINING.md).

## Inference
```bash
python test.py
```
## License
* The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).
* The models weights in this repository are released under the CC-BY-NC 4.0 license as found in the [LICENSE_weights file](LICENSE_weights).


## Reference
We would like to thank the authors of this repo for their contribution.

```
@article{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    year={2023},
    journal={arXiv preprint arXiv:2306.05284},
}
```
