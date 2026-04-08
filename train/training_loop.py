import functools
import os
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler

import sys
[sys.path.append(i) for i in ['../process', '../../ubisoft-laforge-ZeroEGGS-main', '../mydiffusion_zeggs']]
from generate.generate import WavEncoder
from process_zeggs_bvh import pose2bvh

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
import warnings
warnings.filterwarnings('ignore')

class TrainLoop:
    def __init__(self, args, model,model_adv, diffusion,diffusion_hy, device, data=None):
        self.args = args
        self.data = data
        self.model = model
        self.model_adv = model_adv
        self.diffusion = diffusion
        self.diffusion_hy = diffusion_hy
        #self.cond_mode = model.cond_mode
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        # self.save_interval = args.save_interval
        # self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        # self.num_steps = args.num_steps
        self.num_epochs = 40000
        self.n_seed = 8

        self.sync_cuda = torch.cuda.is_available()

        # self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.mp_trainer_adv = MixedPrecisionTrainer(
            model=self.model_adv,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir

        self.device = device
        if args.audio_feat == "wav encoder":
            self.WavEncoder = WavEncoder().to(self.device)
            self.opt = AdamW([
                {'params': self.mp_trainer.master_params, 'lr':self.lr, 'weight_decay':self.weight_decay},
                {'params': self.WavEncoder.parameters(), 'lr':self.lr}
            ])
        elif args.audio_feat == "mfcc" or args.audio_feat == 'wavlm':
            self.opt = AdamW([
                {'params': self.mp_trainer.master_params, 'lr':self.lr, 'weight_decay':self.weight_decay}
            ])
            self.opt_adv = AdamW([
                {'params': self.mp_trainer_adv.master_params, 'lr':1.25e-4, 'weight_decay':self.weight_decay}
            ])

        # if self.resume_step:
        #     self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.schedule_sampler_hy = create_named_schedule_sampler(self.schedule_sampler_type, diffusion_hy)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        # if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
        #     mm_num_samples = 0  # mm is super slow hence we won't run it during training
        #     mm_num_repeats = 0  # mm is super slow hence we won't run it during training
        #     gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
        #                                     split=args.eval_split,
        #                                     hml_mode='eval')
        #
        #     self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
        #                                            split=args.eval_split,
        #                                            hml_mode='gt')
        #     self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, self.device)
        #     self.eval_data = {
        #         'test': lambda: eval_humanml.get_mdm_loader(
        #             model, diffusion, args.eval_batch_size,
        #             gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
        #             args.eval_num_samples, scale=1.,
        #         )
        #     }
        self.use_ddp = False
        self.ddp_model = self.model
        self.ddp_model_adv = self.model_adv
        self.mask_train = (torch.zeros([self.batch_size, 1, 1, args.n_poses]) < 1).to(self.device)
        self.mask_train1 = (torch.zeros([self.batch_size, 1, 1, 88]) < 1).to(self.device)
        self.mask_test = (torch.zeros([1, 1, 1, args.n_poses]) < 1).to(self.device)
        # self.tmp_audio = torch.from_numpy(np.load('tmp_audio.npy')).unsqueeze(0).to(self.device)
        # self.tmp_mfcc = torch.from_numpy(np.load('10_kieks_0_9_16.npz')['mfcc'][:args.n_poses]).to(torch.float32).unsqueeze(0).to(self.device)
        self.mask_local_train = torch.ones(self.batch_size, args.n_poses).bool().to(self.device)
        self.mask_local_test = torch.ones(1, args.n_poses).bool().to(self.device)

    # def _load_and_sync_parameters(self):
    #     resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
    #
    #     if resume_checkpoint:
    #         self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
    #         logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
    #         self.model.load_state_dict(
    #             dist_util.load_state_dict(
    #                 resume_checkpoint, map_location=self.device
    #             )
    #         )

    # def _load_optimizer_state(self):
    #     main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
    #     opt_checkpoint = bf.join(
    #         bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
    #     )
    #     if bf.exists(opt_checkpoint):
    #         logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
    #         state_dict = dist_util.load_state_dict(
    #             opt_checkpoint, map_location=self.device
    #         )
    #         self.opt.load_state_dict(state_dict)

    def run_loop(self):
        body_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,36,37,38,39,63,64,65,66,67,68,69,70,71,72,73,74]#29
        hand_index=[13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62]#46
        for epoch in range(self.num_epochs):
            # print(f'Starting epoch {epoch}')
            # for _ in tqdm(range(10)):     # 4 steps, batch size, chmod 777
            batch_index_s = 0
            for batch in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break
                # if batch_index_s >= 1:
                #     break
                # else:
                #     batch_index_s += 1
                cond_ = {'y':{}}

                # cond_['y']['text'] = ['A person turns left with medium speed.', 'A human goes slowly about 1.5 meters forward.']

                # motion = torch.rand(2, 135, 1, 80).to(self.device)
                # pose_seq, _, style, audio, mfcc, wavlm = batch  # (batch, 240, 135), (batch, 30), (batch, 64000)
                # pose_seq, _, style, _, _, wavlm = batch
                pose_seq, style, wavlm,text = batch
                #print("wavlm shape:",wavlm.shape)
                #print("style shape:",style.shape)
                #motion = pose_seq.permute(0, 2, 1).unsqueeze(2).to(self.device)
                motion=pose_seq.to(self.device)
                tar_pose_body=torch.zeros([pose_seq.shape[0],pose_seq.shape[1],451])
                tar_pose_body[:,:,:13]=pose_seq[:,:,:13]
                tar_pose_body[:,:,13:100]=(pose_seq[:,:,13:238].reshape(pose_seq.shape[0],pose_seq.shape[1],75,3)[:,:,body_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))#lpos 
                tar_pose_body[:,:,100:274]=(pose_seq[:,:,238:688].reshape(pose_seq.shape[0],pose_seq.shape[1],75,2,3)[:,:,body_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))#ltxy 
                tar_pose_body[:,:,274:361]=(pose_seq[:,:,688:913].reshape(pose_seq.shape[0],pose_seq.shape[1],75,3)[:,:,body_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))#lvel 
                tar_pose_body[:,:,361:448]=(pose_seq[:,:,913:1138].reshape(pose_seq.shape[0],pose_seq.shape[1],75,3)[:,:,body_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))#lvrt 
                tar_pose_body[:,:,448:451]=pose_seq[:,:,1138:1141]

                tar_pose_hand=torch.zeros([pose_seq.shape[0],pose_seq.shape[1],690])
                tar_pose_hand[:,:,:138]=(pose_seq[:,:,13:238].reshape(pose_seq.shape[0],pose_seq.shape[1],75,3)[:,:,hand_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))
                tar_pose_hand[:,:,138:414]=(pose_seq[:,:,238:688].reshape(pose_seq.shape[0],pose_seq.shape[1],75,2,3)[:,:,hand_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))
                tar_pose_hand[:,:,414:552]=(pose_seq[:,:,688:913].reshape(pose_seq.shape[0],pose_seq.shape[1],75,3)[:,:,hand_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))
                tar_pose_hand[:,:,552:690]=(pose_seq[:,:,913:1138].reshape(pose_seq.shape[0],pose_seq.shape[1],75,3)[:,:,hand_index,:].reshape(pose_seq.shape[0],pose_seq.shape[1],-1))

                tar_pose_body=tar_pose_body.permute(0, 2, 1).unsqueeze(2).cuda()
                tar_pose_hand=tar_pose_hand.permute(0, 2, 1).unsqueeze(2).cuda()
                #print("motion shape:",motion.shape)pose_seq.permute(0, 2, 1).unsqueeze(2).to(self.device)
                cond_['y']['seed_body'] = tar_pose_body[..., 0:self.n_seed]
                cond_['y']['seed_hand'] = tar_pose_hand[..., 0:self.n_seed]

                #cond_['y']['seed_body_end'] = tar_pose_body[:, -self.n_seed:,:]
                #cond_['y']['seed_hand_end'] = tar_pose_hand[:, -self.n_seed:,:]
                #cond_['y']['seed_body']=torch.cat([cond_['y']['seed_body'],cond_['y']['seed_body_end']],axis=1)
                #cond_['y']['seed_hand']=torch.cat([cond_['y']['seed_hand'],cond_['y']['seed_hand_end']],axis=1)
                # motion = motion[..., self.n_seed:]
                cond_['y']['style'] = style.to(self.device)
                cond_['y']['text'] = text
                #print("text:",text)
                cond_['y']['mask_local'] = self.mask_local_train

                if self.args.audio_feat == 'wav encoder':
                    # cond_['y']['audio'] = torch.rand(240, 2, 32).to(self.device)
                    cond_['y']['audio'] = self.WavEncoder(audio.to(self.device)).permute(1, 0, 2)       # (batch, 240, 32)
                elif self.args.audio_feat == 'mfcc':
                    # cond_['y']['audio'] = torch.rand(80, 2, 13).to(self.device)
                    cond_['y']['audio'] = mfcc.to(torch.float32).to(self.device).permute(1, 0, 2)       # [self.n_seed:, ...]      # (batch, 80, 13)
                elif self.args.audio_feat == 'wavlm':
                    cond_['y']['audio'] = wavlm.to(torch.float32).to(self.device)

                cond_['y']['mask'] = self.mask_train        # [..., self.n_seed:]
                cond_['y']['mask1'] = self.mask_train1

                self.run_step(epoch,motion,tar_pose_body, tar_pose_hand,cond_)
                # if self.step % self.log_interval == 0:
                #     for k,v in logger.get_current().name2val.items():
                #         if k == 'loss':
                #             print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))
                #         elif k=='rot_mse_body':
                #             print('step[{}]: rot_mse_body[{:0.5f}]'.format(self.step+self.resume_step, v))
                #         elif k=='rot_mse_hand':
                #             print('step[{}]: rot_mse_hand[{:0.5f}]'.format(self.step+self.resume_step, v))
                #         elif k=='rot_mse_body_hy':
                #             print('step[{}]: rot_mse_body_hy[{:0.5f}]'.format(self.step+self.resume_step, v))
                #         elif k=='fc':
                #             print('step[{}]: fc[{:0.5f}]'.format(self.step+self.resume_step, v))
                
                # if self.step % 10000 == 0:
                #     sample_fn = self.diffusion.p_sample_loop
                #
                #     model_kwargs_ = {'y': {}}
                #     model_kwargs_['y']['mask'] = self.mask_test     # [..., self.n_seed:]
                #     model_kwargs_['y']['seed'] = torch.zeros([1, 1141, 1, self.n_seed]).to(self.device)
                #     model_kwargs_['y']['style'] = torch.zeros([1, 6]).to(self.device)
                #     model_kwargs_['y']['mask_local'] = self.mask_local_test
                #     if self.args.audio_feat == 'wav encoder':
                #         model_kwargs_['y']['audio'] = self.WavEncoder(self.tmp_audio).permute(1, 0, 2)
                #         # model_kwargs_['y']['audio'] = torch.rand(240, 1, 32).to(self.device)
                #     elif self.args.audio_feat == 'mfcc':
                #         model_kwargs_['y']['audio'] = self.tmp_mfcc.permute(1, 0, 2)        # [self.n_seed:, ...]
                #         # model_kwargs_['y']['audio'] = torch.rand(80, 1, 13).to(self.device)
                #     elif self.args.audio_feat == 'wavlm':
                #         model_kwargs_['y']['audio'] = torch.randn(1, 1, 1024).to(self.device)
                #
                #     sample = sample_fn(
                #         self.model,
                #         (1, 1141, 1, self.args.n_poses),     #  - self.n_seed
                #         clip_denoised=False,
                #         model_kwargs=model_kwargs_,
                #         skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                #         init_image=None,
                #         progress=True,
                #         dump_steps=None,
                #         noise=None,
                #         const_noise=False,
                #     )       # (1, 135, 1, 240)
                #
                #     sampled_seq = sample.squeeze(0).permute(1, 2, 0)
                #     data_mean_ = np.load("../../ubisoft-laforge-ZeroEGGS-main/Data/processed_v1/processed/mean.npz")['mean']
                #     data_std_ = np.load("../../ubisoft-laforge-ZeroEGGS-main/Data/processed_v1/processed/std.npz")['std']
                #
                #     data_mean = np.array(data_mean_).squeeze()
                #     data_std = np.array(data_std_).squeeze()
                #     std = np.clip(data_std, a_min=0.01, a_max=None)
                #     out_poses = np.multiply(np.array(sampled_seq[0].detach().cpu()), std) + data_mean
                #
                #     pipeline_path = '../../../My/process/resource/data_pipe_20_rotation.sav'
                #     save_path = 'inference_zeggs_mymodel3_wavlm'
                #     prefix = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
                #     if not os.path.exists(save_path):
                #         os.mkdir(save_path)
                #     # make_bvh_GENEA2020_BT(save_path, prefix, out_poses, smoothing=False, pipeline_path=pipeline_path)
                #
                #     pose2bvh(out_poses, os.path.join(save_path, prefix + '.bvh'), length=self.args.n_poses)

                if self.step % 10000 == 0:
                    self.save()
                    # self.model.eval()
                    # self.evaluate()
                    # self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        # if (self.step - 1) % 50000 != 0:
        #     self.save()
            # self.evaluate()


    def run_step(self, epoch, batch,batch_body,batch_hand, cond):
        self.forward_backward(epoch,batch,batch_body,batch_hand, cond)      # torch.Size([64, 251, 1, 196]) cond['y'].keys() dict_keys(['mask', 'lengths', 'text', 'tokens'])
        #self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, epoch,batch,batch_body,batch_hand, cond):
        #self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            
            latent_z_body = torch.randn(self.batch_size, 64, device=self.device)
            latent_z_hand = torch.randn(self.batch_size, 64, device=self.device)


            micro = batch
            micro_body = batch_body
            micro_hand = batch_hand
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)
            t_hy, weights_hy = self.schedule_sampler_hy.sample(micro.shape[0], self.device)


            for p in self.mp_trainer_adv.master_params:  
                p.requires_grad = True  
            self.mp_trainer_adv.zero_grad()
            #D loss  backward
            errD_real, errD_real_body, errD_real_hand, D_real_body, D_real_hand, x_t_body, x_t_body_p1, x_t_hand, x_t_hand_p1 = self.diffusion.training_losses_adv(self.ddp_model,self.ddp_model_adv,micro,micro_body,micro_hand,t,t_hy,latent_z_body,latent_z_hand,model_kwargs=micro_cond,dataset='kit')
            if self.step % 20 == 0:
                self.mp_trainer_adv.backward(errD_real,retain_graph=True)
            if self.step % 1 == 0:
                grad_real_body = torch.autograd.grad(
                        outputs=D_real_body.sum(), inputs=x_t_body, create_graph=True
                        )[0]
                grad_penalty_body = (
                            grad_real_body.reshape(grad_real_body.size(0), -1).norm(2, dim=1) ** 2
                            ).mean()
                grad_penalty_body = 0.02 / 2 * grad_penalty_body

                grad_real_hand = torch.autograd.grad(
                        outputs=D_real_hand.sum(), inputs=x_t_hand, create_graph=True
                        )[0]
                grad_penalty_hand = (
                            grad_real_hand.reshape(grad_real_hand.size(0), -1).norm(2, dim=1) ** 2
                            ).mean()
                grad_penalty_hand = 0.02 / 2 * grad_penalty_hand

                grad_penalty = grad_penalty_body + grad_penalty_hand
                if self.step % 20 == 0:
                    self.mp_trainer_adv.backward(grad_penalty)
            # train with fake
            gen_body_pos_sample,gen_hand_pos_sample = self.diffusion.training_losses_gen(self.ddp_model,self.ddp_model_adv,micro,x_t_body_p1.detach(),x_t_hand_p1.detach(),t,t_hy,latent_z_body,latent_z_hand,model_kwargs=micro_cond,dataset='kit')
            errD_fake = self.diffusion.training_losses_adv_fake(self.ddp_model,self.ddp_model_adv,gen_body_pos_sample,x_t_body_p1.detach(),gen_hand_pos_sample,x_t_hand_p1.detach(),t,model_kwargs=micro_cond,dataset='kit')
            if self.step % 20 == 0:
                self.mp_trainer_adv.backward(errD_fake)
            errD = errD_real + errD_fake
            # Update D
            if self.step % 20 == 0:
                self.mp_trainer_adv.optimize(self.opt_adv)

            #update G
            for p in self.mp_trainer_adv.master_params:
                p.requires_grad = False
            self.mp_trainer.zero_grad()
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)
            losses, gen_body_pos_sample, gen_hand_pos_sample = self.diffusion.training_losses(self.ddp_model, micro,micro_body,micro_hand, t,t_hy, model_kwargs=micro_cond, dataset='kit')
            gem_loss = (losses["loss"] * weights).mean()
            AFD_loss = losses["cross_info_loss"]

            errG = self.diffusion.training_losses_adv_gen(self.ddp_model,self.ddp_model_adv,gen_body_pos_sample,x_t_body_p1.detach(),gen_hand_pos_sample,x_t_hand_p1.detach(),t,model_kwargs=micro_cond,dataset='kit')
            if self.step % 20 == 0:
                self.mp_trainer.backward(errG*1.0 + 10.0 * gem_loss + 0.5*AFD_loss)
            else:
                self.mp_trainer.backward(errG*0.0 + 10.0 * gem_loss + 0.5*AFD_loss)   #sddim

            
            self.mp_trainer.optimize(self.opt)
            if self.step % 20 == 0:
                #print('iterations{}, Gem Loss: {}'.format(self.step, gem_loss.item()))
                print('iterations{}, errG Loss: {}, Gem Loss: {}, AFD Loss: {}'.format(self.step, errG.item(), gem_loss.item(), AFD_loss.item()))
                print('iterations{}, D Loss:{} ,errD_real Loss: {}, errD_fake Loss: {} '.format(self.step, errD.item(), errD_real.item(), errD_fake.item()))

                # print('iterations{}, errG Loss: {}, Gem Loss: {}'.format(self.step, errG.item(), gem_loss.item()))
                # print('iterations{}, D Loss:{} ,errD_real Loss: {}, errD_fake Loss: {} '.format(self.step, errD.item(), errD_real.item(), errD_fake.item()))
            # compute_losses = functools.partial(
            #     self.diffusion.training_losses,
            #     self.ddp_model,
            #     micro,  # [bs, ch, image_size, image_size]      # x_start, (2, 135, 1, 240)
            #     micro_body,
            #     micro_hand,
            #     t,  # [bs](int) sampled timesteps
            #     t_hy,
            #     model_kwargs=micro_cond,
            #     dataset='kit'
            # )

            # if last_batch or not self.use_ddp:
            #     losses, gen_body_pos_sample, gen_hand_pos_sample = compute_losses()
            # else:
            #     with self.ddp_model.no_sync():
            #         losses, gen_body_pos_sample, gen_hand_pos_sample = compute_losses()

            # if isinstance(self.schedule_sampler, LossAwareSampler):
            #     self.schedule_sampler.update_with_local_losses(
            #         t, losses["loss"].detach()
            #     )

            #losses, gen_body_pos_sample, gen_hand_pos_sample = self.diffusion.training_losses(self.ddp_model, micro,micro_body,micro_hand, t,t_hy, model_kwargs=micro_cond, dataset='kit')
            # loss = (losses["loss"] * weights).mean()
            # log_loss_dict(
            #     self.diffusion, t, {k: v * weights for k, v in losses.items()}
            # )
            # self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
