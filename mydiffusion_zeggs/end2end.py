import pdb
import logging
logging.getLogger().setLevel(logging.INFO)
from torch.utils.data import DataLoader
from data_loader.lmdb_data_loader import TrinityDataset
import torch
import yaml
from pprint import pprint
from easydict import EasyDict
from configs.parse_args import parse_args
import os
import sys
[sys.path.append(i) for i in ['.', '..', '../model', '../train']]
from utils.model_util import create_gaussian_diffusion
from training_loop import TrainLoop
from model.mdm import MDM
#from model.discriminator import DISCRIMINATOR
from model.semi_discriminator import DISCRIMINATOR
import torch.nn as nn
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip,create_gaussian_diffusion_hy
import warnings
warnings.filterwarnings('ignore')



from transformers import AutoTokenizer, AutoModel
def create_model_and_diffusion(args):
    model = MDM(modeltype='', njoints=1141, nfeats=1, cond_mode = 'cross_local_attention3_style1', action_emb = 'tensor', audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=256, n_seed=8, cond_mask_prob=0.1)
    diffusion = create_gaussian_diffusion()
    diffusion_hy = create_gaussian_diffusion_hy()
    return model, diffusion,diffusion_hy

def discriminator_model(args):
    model = DISCRIMINATOR(modeltype='', njoints=1141, nfeats=1, cond_mode = 'cross_local_attention3_style1', action_emb = 'tensor', audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=256, n_seed=8, cond_mask_prob=0.1)
    return model

def main(args, device):
    # dataset
    print("args.n_poses:",args.n_poses)
    train_dataset = TrinityDataset(args.train_data_path,
                                   n_poses=args.n_poses,
                                   subdivision_stride=args.subdivision_stride,
                                   pose_resampling_fps=args.motion_resampling_framerate, model='WavLM', device=device)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)

    val_dataset = TrinityDataset(args.val_data_path,
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate, model='WavLM', device=device)
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=False)

    logging.info('len of train loader:{}, len of test loader:{}'.format(len(train_loader), len(test_loader)))

    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    model, diffusion,diffusion_hy = create_model_and_diffusion(args)
    '''
    model_path='/apdcephfs/share_1290939/shaolihuang/ykcheng/DiffCoSG/new3/DiffuseStyleGesture/main/mydiffusion_zeggs/beats_1/model000780000.pt'
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    '''
    #model.load_state_dict(new_state_dict)
    #model = nn.DataParallel(model, device_ids=device_ids,output_device=device_ids[-1])
    model.to(mydevice)
    model.train()

    #对抗器
    model_adv = discriminator_model(args)

    model_adv.to(mydevice)
    model_adv.train()

    #model = nn.DataParallel(model, device_ids=device_ids,output_device=device_ids[-1])
    #model.to(mydevice)
    TrainLoop(args, model,model_adv, diffusion,diffusion_hy, mydevice, data=train_loader).run_loop()


if __name__ == '__main__':
    '''
    cd mydiffusion_zeggs/
    '''

    args = parse_args()
    
    mydevice = torch.device('cuda:' + args.gpu)
    torch.cuda.set_device(int(args.gpu))
    
    '''
    device_ids = [1,2,3,4]
    mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_ids[0])
    '''

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)

    main(config, mydevice)
