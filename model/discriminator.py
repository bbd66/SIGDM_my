import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
from local_attention import LocalAttention
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModel


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class WavEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_feature_map = nn.Linear(1024, 64)

    def forward(self, rep):
        rep = self.audio_feature_map(rep)
        return rep



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)      # (5000, 128)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)     # (5000, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class DISCRIMINATOR(nn.Module):
    def __init__(self, modeltype, njoints, nfeats,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, audio_feat='', n_seed=1, cond_mode='', **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0

        self.audio_feat = audio_feat
        if audio_feat == 'wav encoder':
            self.audio_feat_dim = 32
        elif audio_feat == 'mfcc':
            self.audio_feat_dim = 13
        elif self.audio_feat == 'wavlm':
            print('USE WAVLM')
            self.audio_feat_dim = 64        # Linear 1024 -> 64
            self.WavEncoder = WavEncoder()
            #self.WavEncoder = AudioEncoder()

        self.sequence_pos_encoder = PositionalEncoding(int(self.latent_dim/2), self.dropout)
        self.emb_trans_dec = emb_trans_dec

        self.cond_mode = cond_mode
        self.num_head = 8

        self.embed_timestep = TimestepEmbedder(int(self.latent_dim / 2), self.sequence_pos_encoder)

        self.n_seed = n_seed

        self.input_process_body = InputProcess(self.data_rep, 451  + self.gru_emb_dim, self.latent_dim)
        self.input_process_hand = InputProcess(self.data_rep, 690  + self.gru_emb_dim, self.latent_dim)

        self.embed_text_body = nn.Linear(451 * n_seed, int(self.latent_dim / 2))
        self.embed_text_hand = nn.Linear(690 * n_seed, int(self.latent_dim / 2))


        self.latent_dim_body = 2 * 451 * 88 + 128 + 88*64
        self.latent_dim_hand = 2 * 690 * 88 + 128 + 88*64
        # 第一个块：Linear Block + SELU
        self.block1_body = nn.Sequential(
            nn.Linear(self.latent_dim_body, 1024),
            nn.SELU()
        )

        # 第二个块：Linear Block + Group Norm + SELU
        self.block2_body = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GroupNorm(32, 1024),
            nn.SELU()
        )

        # 第三个块：Linear Block + SELU
        self.block3_body = nn.Sequential(
            nn.Linear(1024, 256),
            nn.SELU()
        )

        # 第四个块：Linear Block + Group Norm + SELU
        self.block4_body = nn.Sequential(
            nn.Linear(256, 256),
            nn.GroupNorm(32, 256),
            nn.SELU()
        )

        # 第五个块：Linear Block + SELU
        self.block5_body = nn.Sequential(
            nn.Linear(256, 64),
            nn.SELU()
        )

        self.block6_body = nn.Sequential(
            nn.Linear(64, 64),
            nn.GroupNorm(16, 64),
            nn.SELU()
        )

        # 第六个块：Linear Block
        self.block7_body = nn.Sequential(
            nn.Linear(64, 1)
        )


        # 第一个块：Linear Block + SELU
        self.block1_hand = nn.Sequential(
            nn.Linear( self.latent_dim_hand, 1024),
            nn.SELU()
        )

        # 第二个块：Linear Block + Group Norm + SELU
        self.block2_hand = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GroupNorm(32, 1024),
            nn.SELU()
        )

        # 第三个块：Linear Block + SELU
        self.block3_hand = nn.Sequential(
            nn.Linear(1024, 256),
            nn.SELU()
        )

        # 第四个块：Linear Block + Group Norm + SELU
        self.block4_hand = nn.Sequential(
            nn.Linear(256, 256),
            nn.GroupNorm(32, 256),
            nn.SELU()
        )

        # 第五个块：Linear Block + SELU
        self.block5_hand = nn.Sequential(
            nn.Linear(256, 64),
            nn.SELU()
        )

        # 第四个块：Linear Block + Group Norm + SELU
        self.block6_hand = nn.Sequential(
            nn.Linear(64, 64),
            nn.GroupNorm(16, 64),
            nn.SELU()
        )

        # 第六个块：Linear Block
        self.block7_hand = nn.Sequential(
            nn.Linear(64, 1)
        )
        

    def forward(self, x_body_cp, x_body_p1_cp,x_hand_cp, x_handp1_cp, timesteps, y=None,uncond_info=False):
        x_body = x_body_cp.clone() 
        x_body_p1 = x_body_p1_cp.clone() 
        x_hand = x_hand_cp.clone() 
        x_handp1 = x_handp1_cp.clone() 


        bs, njoints_body, nfeats_body, nframes = x_body.shape
        bs, njoints_hand, nfeats_hand, nframes = x_hand.shape

        force_mask=uncond_info
        

        emb_t=self.embed_timestep(timesteps)#bs,128

        if self.audio_feat == 'wavlm':
            enc_text = self.WavEncoder(y['audio']).permute(1, 0, 2)#64
        else:
            enc_text = y['audio']

        embed_text_body = self.embed_text_body(y['seed_body'].squeeze(2).reshape(bs, -1))       # (bs, 128)
        embed_text_hand = self.embed_text_hand(y['seed_hand'].squeeze(2).reshape(bs, -1))       # (bs, 128)

        embed_style_2_body = (emb_t + embed_text_body).reshape(bs,-1)
        embed_style_2_hand = (emb_t + embed_text_hand).reshape(bs,-1)
        

        x_body = x_body.reshape(bs, njoints_body * nfeats_body, 1, nframes)
        x_body_p1 = x_body_p1.reshape(bs, njoints_body * nfeats_body, 1, nframes)
        x_hand = x_hand.reshape(bs, njoints_hand * nfeats_hand, 1, nframes)
        x_handp1 = x_handp1.reshape(bs, njoints_hand * nfeats_hand, 1, nframes)

        x_body = x_body.reshape(bs,-1)
        x_body_p1 = x_body_p1.reshape(bs,-1)
        x_hand = x_hand.reshape(bs,-1)
        x_handp1 = x_handp1.reshape(bs,-1)

        enc_text = enc_text.reshape(bs,-1)

        #print("embed_style_2_body shape:",embed_style_2_body.shape)
        xseq_body = torch.cat([x_body,x_body_p1,embed_style_2_body,enc_text],axis=1)
        xseq_hand = torch.cat([x_hand,x_handp1,embed_style_2_hand,enc_text],axis=1)


        xseq_body = self.block1_body(xseq_body)
        xseq_body = self.block2_body(xseq_body)
        xseq_body = self.block3_body(xseq_body)
        xseq_body = self.block4_body(xseq_body)
        xseq_body = self.block5_body(xseq_body)
        xseq_body = self.block6_body(xseq_body)
        output_body = self.block7_body(xseq_body)

        xseq_hand = self.block1_hand(xseq_hand)
        xseq_hand = self.block2_hand(xseq_hand)
        xseq_hand = self.block3_hand(xseq_hand)
        xseq_hand = self.block4_hand(xseq_hand)
        xseq_hand = self.block5_hand(xseq_hand)
        xseq_hand = self.block6_hand(xseq_hand)
        output_hand = self.block7_hand(xseq_hand)

        return output_body, output_hand




