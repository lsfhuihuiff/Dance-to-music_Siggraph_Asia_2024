import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import torch
import random
import numpy as np

torch.cuda.set_device(1)
class Audio:
    def __init__(self, path):
        self.path = path

def initialize_seed(seed):
    random.seed(seed)


    np.random.seed(seed)


    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    seed = 2512
    initialize_seed(seed)
    model = MusicGen.get_pretrained('path/to/pretrained/model')
    model.set_generation_params(duration=5.12) 

    embedding_manager_path = 'path/to/encoder/branch/model'
    model.lm.condition_provider.conditioners.description.embedding_manager.load(embedding_manager_path)
    save_dir = 'path/to/generate_wav'
    text = "a @ music with * as the rhythm"
    audio_dir = 'path/to/real_wav'
    for subdir, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(subdir, file)
                    description = [[text, Audio("")]]
                    description[0][1].path = audio_path
                    wav = model.generate(description)
                    rel_dir = os.path.relpath(subdir, audio_dir)
                    if not os.path.exists(os.path.join(save_dir, rel_dir)):
                        os.makedirs(os.path.join(save_dir, rel_dir))
                    save_path = os.path.join(save_dir, rel_dir, file)
                    for idx, one_wav in enumerate(wav):
                        audio_write(save_path[:-4], one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

                    print(file)       
