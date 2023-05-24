import os
import SimpleITK as sitk
import numpy as np
import torch
import pprint
import wandb

from utils import general
from models import models
from utils.metric import measure_metrics
from tmp.IDIR.utils.general import load_image_BraTS, load_image_camcan
from objectives.dmmr import DMMRLoss

# data_dir = "/vol/alan/projects/BraTS/MICCAI_BraTS2020_TrainingData/"
data_dir = "/vol/alan/projects/camcan_malpem/train/"
mask_dir = "/vol/alan/users/toz/midir-pycharm/datasets/camcan_masks/"
out_dir = "../idir_out/"
mod = 't1t2'  # mod = 't1t1'
fixed_id = "CC110037"
moving_id = "CC110062"

(
    fixed_img,
    moving_img,
    fixed_mask,
    moving_mask,
    fixed_sitk,
    moving_sitk,
    fixed_seg_mask,
    moving_seg_mask,
) = load_image_camcan(fixed_id, moving_id, data_dir, mask_dir, mod)

kwargs = {}
kwargs["verbose"] = True
kwargs["hyper_regularization"] = False
kwargs["jacobian_regularization"] = False
kwargs["bending_regularization"] = True
kwargs["network_type"] = "SIREN"  # Options are "MLP" and "SIREN"
save_folder = os.path.join(str(out_dir), f'registration-{fixed_id}-{moving_id}-{mod}')
kwargs["save_folder"] = save_folder
kwargs["mask"] = fixed_mask
kwargs['target_seg'] = fixed_seg_mask
kwargs['source_seg'] = moving_seg_mask
kwargs["image_shape"] = (189, 233, 197)  # (155, 240, 240) for brats
kwargs["epochs"] = 2500
kwargs["log_interval"] = 200
# loss = 'ncc'
# loss = 'mse'
# loss = 'l1'
# loss = 'smoothl1'
# loss = 'huber'
# loss = "nmi"  # loss is always NaN for MIGaussian loss -> why?
loss = "mi"
# loss = "dmmr"
kwargs['loss_function'] = loss

log = True

ImpReg = models.ImplicitRegistrator(fixed_img, moving_img, **kwargs)

if log:
    run = wandb.init(project="IDIR", config={**ImpReg.args, **kwargs})

ImpReg.fit()

# Get transformed moving image from trained registration model
moving_coord_tensor = general.make_coordinate_tensor(ImpReg.moving_image.shape).cpu()
net = ImpReg.network.cpu()
output = net(moving_coord_tensor)
transformed_moving_image = ImpReg.transform(output,
                                            moving_coord_tensor,
                                            ImpReg.moving_image.cpu())

print(f"moving_coord_tensor.shape -> {moving_coord_tensor.shape}")
print(f"output.shape -> {output.shape}")
print(f"transformed_moving_image.shape -> {transformed_moving_image.shape}")

disp_pred = output.unsqueeze(0).cpu().detach().numpy()
disp_pred = disp_pred.reshape(1, 3, *ImpReg.moving_image.shape)  # (1, 3, *sizes)

target_pred = transformed_moving_image.detach().numpy()
target_pred = target_pred.reshape(ImpReg.moving_image.shape)
target_pred = np.expand_dims(target_pred, axis=(0, 1))  # (1, 1, *sizes)

target = ImpReg.fixed_image.cpu().detach().numpy()
target = np.expand_dims(target, axis=(0, 1))  # (1, 1, *sizes)

target_seg = sitk.GetArrayFromImage(fixed_seg_mask)
target_seg = np.expand_dims(target_seg, axis=(0, 1))  # (1, 1, *sizes)

source_seg = sitk.GetArrayFromImage(moving_seg_mask)
source_seg = np.expand_dims(source_seg, axis=(0, 1))  # (1, 1, *sizes)

transformed_source_seg = ImpReg.transform(output.cpu(),
                                          moving_coord_tensor.cpu(),
                                          torch.tensor(source_seg).squeeze().cpu())

warped_source_seg = transformed_source_seg.detach().numpy()
warped_source_seg = warped_source_seg.reshape(ImpReg.moving_image.shape)
warped_source_seg = np.expand_dims(warped_source_seg, axis=(0, 1))  # (1, 1, *sizes)

metric_data = {'disp_pred': disp_pred,
               'target': target,
               'target_pred': target_pred,
               'target_seg': target_seg,
               'source_seg': source_seg,
               'warped_source_seg': warped_source_seg,
               }
# metric_data['roi_mask'] = fixed_mask  # check shape of fixed mask, should be (N, 1, *sizes)

# print('datalosslist', ImpReg.data_loss_list) # NaNs
# print('losslist', ImpReg.loss_list) # Not NaNs

if kwargs['verbose'] and log:
    for l in ImpReg.data_loss_list:
        wandb.log({f"loss": l})
    for l in ImpReg.loss_list:
        wandb.log({f"total_loss": l})
    for md in ImpReg.metric_history:
        for k, v in md.items():
            wandb.log({f"{k}": v, 'epoch': md['epoch']})

metric_results = measure_metrics(metric_data, metric_groups=["disp_metrics", "image_metrics", "seg_metrics"])
pprint.pprint(metric_results)

# Saving the images to disk
out_moving_sitk = sitk.GetImageFromArray(np.squeeze(target_pred))
out_moving_sitk.CopyInformation(moving_sitk)
sitk.WriteImage(fixed_sitk, f'{save_folder}/fixed_image_{fixed_id}_{moving_id}_{mod}.nii.gz')
sitk.WriteImage(moving_sitk, f'{save_folder}/moving_image_{fixed_id}_{moving_id}_{mod}.nii.gz')
sitk.WriteImage(out_moving_sitk, f'{save_folder}/transformed_moving_image_{fixed_id}_{moving_id}_{mod}_{loss}.nii.gz')

# ImpReg.moving_image.shape -> torch.Size([155, 240, 240])
# moving_coord_tensor.shape -> torch.Size([8870400, 3]) -> 8870400 = 155 * 240 * 240
# output.shape -> torch.Size([8870400, 3])
# transformed_moving_image.shape -> torch.Size([8870400]) -> reshape it back to (155, 240, 240)

# masked_moving_coord_tensor.shape -> torch.Size([1539196, 3])
# masked_output.shape -> torch.Size([1539196, 3])
# transformed_masked_moving_image.shape -> torch.Size([1539196])
# TODO: figure out how to get the transformed masked image back to the original shape

# masked_moving_coord_tensor = moving_coord_tensor[moving_mask.flatten() > 0, :]
# masked_output = net(masked_moving_coord_tensor)
# transformed_masked_moving_image = ImpReg.transform(masked_output,
#                                                    masked_moving_coord_tensor,
#                                                    ImpReg.moving_image.cpu())
# print(f"masked_moving_coord_tensor.shape -> {masked_moving_coord_tensor.shape}")
# print(f"transformed_masked_moving_image.shape -> {transformed_masked_moving_image.shape}")
