import json
import argparse
from generation.load_model import get_models

import numpy as np
import torch
from utils.trajectory_plot import draw_circle_with_waves, draw_circle_with_waves2, draw_straight_line
from exit.utils import visualize_2motions
from utils.mint import convert_ControlMM_to_3d, save_to_visualizer
import os
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--iter_each', type=int, default=100, help='number of logits optimization at each unmask step') 
parser.add_argument('--iter_last', type=int, default=600, help='number of logits optimization at the last unmask step') 
parser.add_argument('--path_name', type=str, default='./output/test')
parser.add_argument('--save_to_visualizer', type=str, default=None, help='path to save the visualizer output')
parser.add_argument('--out_name', type=str, default='mask_control_out.json', help='name of the output file for visualizer')
parser.add_argument('--postfix', type=str, default='', help='postfix for the output file name')
parser.add_argument('--overwrite', action='store_true', help='overwrite existing files', default=False)
parser.add_argument('--show', action='store_true', help='auto run html') 
parser.add_argument('--target_motion', type=str, required=True, help='path to input motion file')
args = parser.parse_args()

os.makedirs(f'{args.path_name}/source', mode=os.umask(0), exist_ok=args.overwrite)

with open("./generation/args.json", "r") as f:
    opt = json.load(f)
opt = argparse.Namespace(**opt)

# moments = [mean, std] each has shape [1, 263]
ct2m_transformer, vq_model, res_model, moment = get_models(opt)
traj1 = draw_circle_with_waves()
traj2 = draw_circle_with_waves2()

clip_text = ['a person walks then jumps']

# m_length = torch.tensor([196, 196, 196]).cuda()
m_length = torch.tensor([196]).cuda()

k=0
# global_joint = torch.zeros((m_length.shape[0], 196, 22, 3), device=m_length.device) # B x T x 22 x 3
global_joint = torch.zeros((m_length.shape[0], m_length[k], 22, 3), device=m_length.device) # B x T x 22 x 3
target_motion = np.load(args.target_motion)  # Load input motion from file
target_motion = torch.tensor(target_motion, dtype=torch.float32).to(m_length.device)
target_motion = target_motion[None, :m_length[k], :, :]  # Ensure shape is B x T x J x D

# global_joint[k, :, 0] = traj1
# global_joint[k, :, 20] = traj2
if target_motion.shape[1] < m_length[k]:
    global_joint[k, :target_motion.shape[1], :, :] = target_motion[k, :, :, :]
else:
    global_joint[k, :m_length[k]] = target_motion[k, :m_length[k]]

traj = draw_straight_line(step_length=0.01) # T x 3
# global_joint[k, :m_length[k], 0, 0] += traj[:m_length[k], 0]  # Set x-coordinates of the first joint
# global_joint[k, :m_length[k], 0, 2] += traj[:m_length[k], 2]  # Set z-coordinates of the first joint
traj = traj[:, None, :]
global_joint[k, :m_length[k]] += traj[:m_length[k]]  # Set z-coordinates of the first joint
global_joint_mask = (global_joint.sum(-1) != 0) # B x T x 22    # True is edited, False is not edited
# global_joint_mask[k, :m_length[k], 0] = False  # Ensure the first joint is always considered edited

print(' Optimizing...')
# pred_motions_denorm = B x 196 x 22 x 3
# pred_motions = B x 196 x 263
pred_motions_denorm, pred_motions = ct2m_transformer.generate_with_control(clip_text, m_length, time_steps=10, cond_scale=4,
                                                                        temperature=1, topkr=.9,
                                                                        force_mask=opt.force_mask, 
                                                                        vq_model=vq_model, 
                                                                        global_joint=global_joint, 
                                                                        global_joint_mask=global_joint_mask,
                                                                        _mean=torch.tensor(moment[0]).cuda(),
                                                                        _std=torch.tensor(moment[1]).cuda(),
                                                                        res_cond_scale=5,
                                                                        res_model=res_model,
                                                                        control_opt = {
                                                                            'each_lr': 6e-2,
                                                                            'each_iter': args.iter_each,
                                                                            'lr': 6e-2,
                                                                            'iter': args.iter_last,
                                                                        })
print('Done.')


# path 1
r_pos = pred_motions_denorm[k, :m_length[k], 0]
root_path = r_pos.detach().cpu().numpy()

# path 2
root_path2 = pred_motions_denorm[k, :, 0, :m_length[k]].detach().cpu().numpy()

pred_motions_3d = convert_ControlMM_to_3d(
    pred_motions = pred_motions[k].detach().cpu().numpy(),
    moment = moment,
    length = m_length[k],
    dataset_name='t2m'
)
data_dict = {
    'motion': pred_motions_3d[None].transpose(0, 2, 3, 1),    # B x T x 22 x 3 -> B x 22 x 3 x T
    'text': clip_text,
    'condition_trajectory': global_joint[k][:m_length[k]].detach().cpu().numpy()[None],   # B x T x 22 x 3
    'condition_mask': global_joint_mask[k][:m_length[k]].detach().cpu().numpy()[None],  # B x T x 22
}
out_name = f'{args.out_name}.json' if not args.postfix else f'{args.out_name}_{args.postfix}.json'
save_to_visualizer(data_dict, save_dir=args.save_to_visualizer, out_name=out_name)

visualize_2motions(pred_motions[k].detach().cpu().numpy(), 
                moment[1], 
                moment[0], 
                't2m', 
                m_length[k], 
                # pred_motions[k].detach().cpu().numpy(),
                root_path=traj1.detach().cpu().numpy(),
                root_path2=traj2.detach().cpu().numpy(),
                save_path=f'{args.path_name}/generation.html',
                show=args.show
                )
np.save(f'{args.path_name}/generation.npy', pred_motions[k, :m_length[0]].detach().cpu().numpy())
np.save(f'{args.path_name}/trj_cond.npy', global_joint[k, :m_length[0]].detach().cpu().numpy())