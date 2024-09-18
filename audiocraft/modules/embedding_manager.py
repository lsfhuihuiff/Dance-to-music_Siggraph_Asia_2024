import torch
from torch import nn
import os
import random

from transformers import CLIPTokenizer
from functools import partial
import numpy as np
from .attention import CrossAttention
import PIL
from PIL import Image
import math
import torch.nn.functional as F
from ..clap import laion_clap
import pdb
import pickle



DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]

def get_t5_token_for_string(tokenizer, string):
    token = tokenizer(string)#"input_ids", "attention_mask"
    token = torch.tensor(token["input_ids"], dtype=torch.int64)
    assert torch.count_nonzero(token) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0]

    return token

def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0, 0]


class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_words=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            model_path=None,
            **kwargs
    ):
        super().__init__()

        self.string_to_token_dict = {}
        
        self.string_to_param_dict = nn.ParameterDict()

        self.initial_embeddings = nn.ParameterDict() # These should not be optimized

        self.progressive_words = progressive_words
        self.progressive_counter = 0

        self.max_vectors_per_token = num_vectors_per_token

     
         # using T5 encoder
        self.is_clip = False
        get_token_for_string = partial(get_t5_token_for_string, embedder[0])
        get_embedding_for_tkn = embedder[1].encoder.embed_tokens
        token_dim = 768
        # if per_image_tokens:
        #     placeholder_strings.extend(per_img_token_list)

        self.poseEncoder = PoseEncoder( max_position=303, hidden_size=768, n_heads=8, d_head=64, dropout = 0.05)
        self.audioEncoder = AudioEncoder(hidden_size=768)
        for idx, placeholder_string in enumerate(placeholder_strings):
            token = get_token_for_string(placeholder_string)

            if initializer_words and idx < len(initializer_words):
                init_word_token = get_token_for_string(initializer_words[idx])

                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn(init_word_token.cpu())


            self.string_to_token_dict[placeholder_string] = token

        
        if model_path is not None:
            self.load(model_path)

    def forward(
            self,
            tokenized_text,
            embedded_text,
            wav_path = None,
            timestep=None
    ):
        # pdb.set_trace()
   
        keypoints_path = wav_path.replace(".wav", ".pkl")
        keypoints_path = keypoints_path.replace("/audio_clips/", "/keypoints_clips/")

        with open(keypoints_path, 'rb') as pkl_file:
            keypoints = pickle.load(pkl_file)
    
        keypoints = torch.from_numpy(keypoints)
        keypoints = keypoints.to(device=embedded_text.device, dtype=embedded_text.dtype)
        keypoints = keypoints.unsqueeze(0)

        b, n, device = *tokenized_text.shape, tokenized_text.device
        for placeholder_string, placeholder_token in self.string_to_token_dict.items():
            # print("placeholeder:",placeholder_string)
            if placeholder_string == "*":
                # placeholder_embedding = self.string_to_param_dict[placeholder_string].to(device)
                placeholder_embedding = self.poseEncoder(keypoints)
            elif placeholder_string == "@":
                try:
                    placeholder_embedding = self.audioEncoder(wav_path, device)  
                except:
                    print("no @")

            if self.max_vectors_per_token == 1:  # If there's only one vector per token, we can do a simple replacement
                placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
                # print("placeholder:", placeholder_idx)
                try:
                    embedded_text[placeholder_idx] = placeholder_embedding.float()
                except:
                    print("unconditional")
            else:  # otherwise, need to insert and keep track of changing indices
                if self.progressive_words:
                    self.progressive_counter += 1
                    max_step_tokens = 1 + self.progressive_counter // PROGRESSIVE_SCALE
                else:
                    max_step_tokens = self.max_vectors_per_token

                num_vectors_for_token = min(placeholder_embedding.shape[0], max_step_tokens)

                placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))

                if placeholder_rows.nelement() == 0:
                    continue

                sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                sorted_rows = placeholder_rows[sort_idx]

                for idx in range(len(sorted_rows)):
                    row = sorted_rows[idx]
                    col = sorted_cols[idx]

                    new_token_row = torch.cat([tokenized_text[row][:col], placeholder_token.repeat(num_vectors_for_token).to(device), tokenized_text[row][col + 1:]], axis=0)[:n]
                    new_embed_row = torch.cat([embedded_text[row][:col], placeholder_embedding[:num_vectors_for_token], embedded_text[row][col + 1:]], axis=0)[:n]

                    embedded_text[row]  = new_embed_row
                    tokenized_text[row] = new_token_row

        return embedded_text


    
        
    def save(self, ckpt_path):
        torch.save({
                "string_to_token": self.string_to_token_dict,
                "poseEncoder": self.poseEncoder,
                "audioEncoderP": self.audioEncoder.projector
                }, ckpt_path)
        
    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print('find keys:', ckpt.keys())
        poseEncoder = {}
        audioEncoderP = {}
        for key in ckpt["poseEncoder"].keys():
            name = key.replace("condition_provider.conditioners.description.embedding_manager.poseEncoder.", "")
            poseEncoder[name] = ckpt["poseEncoder"][key]
        for key in ckpt["audioEncoderP"].keys():
            name = key.replace("condition_provider.conditioners.description.embedding_manager.audioEncoder.projector.", "")
            audioEncoderP[name] = ckpt["audioEncoderP"][key]
        self.poseEncoder.load_state_dict(poseEncoder)
        self.audioEncoder.projector.load_state_dict(audioEncoderP)


    def get_embedding_norms_squared(self):
        all_params = torch.cat(list(self.string_to_param_dict.values()), axis=0) # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)              # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        return self.poseEncoder.parameters()
    def list_embedding_parameters(self):
     
        return list(self.poseEncoder.parameters()) + list(self.audioEncoder.parameters())
        

    def embedding_to_coarse_loss(self):
        
        loss = 0.
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss
    
class Attentions(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim))

    def forward(self, x, context=None):
        x_1 = self.attn1(x) + x
        x_2 = self.attn2(x_1, x) + x_1
        x_3 = self.net(x_2)
        return x_3
    
class AudioEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        for param in self.projector.parameters():
            param.requires_grad = True

    
    def forward(self, audio_path, device):
        data_npy = os.path.dirname(os.path.split(audio_path)[0])
        data_npy = os.path.join(data_npy, "genre.npy")
        data_dict = np.load(data_npy, allow_pickle=True).item()
        audio_embed = data_dict[audio_path]
        audio_embed = torch.tensor(audio_embed).to(device)
        audio_embed = self.projector(audio_embed.unsqueeze(0).unsqueeze(0).float())
        return audio_embed 



class PoseEncoder(nn.Module):
    def __init__(self,  hidden_size, max_position, n_heads, d_head, dropout=0., context_dim=None):
        super().__init__()
        self.rencoder = RhythmEncoder()
        self.pos_emb = nn.Embedding(max_position, 1)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(303),
            nn.Linear(303, hidden_size),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # self.rembed = nn.Sequential(
        #     nn.LayerNorm(hidden_size),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size)
        # )
        self.rembed = Attentions(
                dim=hidden_size, n_heads=n_heads, d_head=d_head, dropout=0.05
            )
        
    def forward(self, keypoints):
 
        b, n, _, _ = keypoints.shape
        cond_rhy_peak, _ = self.rencoder(keypoints)
        position_ids = torch.arange(n-2).unsqueeze(0).to(cond_rhy_peak.device)
        pos_emb = self.pos_emb(position_ids)
        x = self.input_projection(cond_rhy_peak.unsqueeze(1).float() + pos_emb.permute(0, 2, 1))

        x = self.rembed(x)
        return x

class RhythmEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.nbins = 10
        self.win_mean = 2 + 2
        self.win_max = 3 + 3 
        self.threshold = 0.1

    def directogram(self, pose):
        gxy = pose[:, :, :, :2]                     # bs, T, 17, 2, remove confidence
        gxy = gxy.permute(0, 1, 3, 2)               # bs, T, 2, 17
        magnitude = gxy.norm(dim=2)[:, :, None, :]
        phase = torch.atan2(gxy[:, :, 1, :], gxy[:, :, 0, :])
        phase_int = phase * (180 / math.pi) % 180
        phase_int = phase_int[:, :, None, :]
        phase_bin = phase_int.floor().long() % self.nbins
        n, t, c, j = gxy.shape
        out = torch.zeros((n, t, self.nbins, j), dtype=torch.float, device=gxy.device)
        out.scatter_(2, phase_bin, magnitude)
        out = out.sum(3)                                # bs, T, nbins
        return out

    def pick_peak(self, rhy_env):
        bs, n = rhy_env.shape
        rhy_local_mean = rhy_env.unfold(1, self.win_mean, 1).mean(dim=2)
        rhy_local_mean = F.pad(rhy_local_mean, (0, n-rhy_local_mean.size(1)))
        rhy_local_max = torch.max(rhy_env.unfold(1, self.win_max, 1), dim=2)[0]
        rhy_local_max = F.pad(rhy_local_max, (0, n-rhy_local_max.size(1)))
        rhy_global_max = torch.mean(rhy_env, dim=1, keepdim=True).repeat(1, n)
        rhy_peak = ((rhy_local_max - rhy_local_mean) > (0.1 * rhy_global_max)) * (rhy_local_max == rhy_env)
        rhy_peak = rhy_peak.long()
        rhy_peak_mask = F.pad(rhy_peak[:, 1:] - rhy_peak[:, :-1], (0, 1))
        rhy_peak_mask = rhy_peak_mask.bool()
        rhy_peak *= rhy_peak_mask
        return rhy_peak

    def forward(self, pose):
        '''
        input: bs, context_length, 17, 3
        output: rhy_peak: bs, context_length; rhy_env: bs, context_length
        '''
        bs = pose.size(0)                                               # bs, context_length, 17, 3
        motion = pose[:, 1:] - pose[:, :-1]
        # motion = F.pad(motion, (0, 0, 0, 0, 0, 1), mode='constant')   # bs, context_length, 17, 3   
        directo = self.directogram(motion)                              # bs, context_length, K
        sf = directo[:, 1:] - directo[:, :-1]                           # compute spectral flux
        # sf = F.pad(sf, (0, 0, 0, 1), mode="constant")
        # sf_abs = (torch.abs(sf)-sf) / 2 
        sf_abs = (sf + torch.abs(sf)) / 2   
        # sf_abs = torch.abs(sf)                            # only consider increase
        rhy_env = torch.sum(sf_abs, dim=2, keepdim=False)               # bs, context_length
        rhy_env = rhy_env / torch.max(rhy_env, dim=1, keepdim=True)[0]  # normalize to 0-1
        rhy_peak = self.pick_peak(rhy_env)
        return rhy_peak, rhy_env.unsqueeze(-1)
