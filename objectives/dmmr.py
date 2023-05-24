import torch
import torch.nn as nn
import torchio as tio
import omegaconf

from torch.nn.modules.loss import _Loss
from pathlib import Path

import importlib.util

# Specify the path to the module
module_path = "/vol/alan/users/toz/midir-pycharm/model/dmmr.py"

# Create a module name
module_name = "dmmr_module"

# Load the module
spec = importlib.util.spec_from_file_location(module_name, module_path)
dmmr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dmmr_module)

DeepMetricModel = dmmr_module.DeepMetricModel

class DMMRLoss(_Loss):
    def __init__(self,
                 patch_size=17,
                 patch_overlap=4,
                 inference_patch_batch_size=256,
                 base_model_path='/vol/alan/users/toz/midir-pycharm/outputs/2023-04-26/18-55-47/',
                 ):
        super(DMMRLoss, self).__init__()
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.base_model_path = Path(base_model_path)
        self.weights_path = self.base_model_path / 'checkpoints/last.ckpt'
        self.cfg = omegaconf.OmegaConf.load(self.base_model_path / '.hydra/config.yaml')
        self.model = DeepMetricModel(self.cfg).load_from_checkpoint(self.weights_path)
        self.model.freeze()

    def forward(self, fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:

        # TODO: how to deal with the fact that during training it just takes a
        # random sample of points, and my loss takes into account whole images?
        import math

        def find_closest_factors(num):
            factors = []
            for i in range(1, int(math.sqrt(num)) + 1):
                if num % i == 0:
                    factors.append((i, num // i))

            closest_factors = min(factors, key=lambda x: abs(x[0] - x[1]))
            return closest_factors

        def reshape_flattened_tensor(flattened_tensor):
            total_elements = flattened_tensor.numel()
            closest_factors = find_closest_factors(total_elements)
            dim1, dim2 = closest_factors
            dim3 = total_elements // (dim1 * dim2)
            reshaped_tensor = flattened_tensor.view(dim1, dim2, dim3)
            return reshaped_tensor

        fixed = reshape_flattened_tensor(fixed).unsqueeze(0)
        warped = reshape_flattened_tensor(warped).unsqueeze(0)

        subj = tio.Subject(
            mod1=tio.ScalarImage(tensor=fixed),
            mod2=tio.ScalarImage(tensor=warped),
        )

        return self.process_subj(subj)

    def process_subj(self, subj):
        grid_sampler = tio.inference.GridSampler(
            subj,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap,
        )
        patch_loader = torch.utils.data.DataLoader(grid_sampler,
                                                   batch_size=1)
        aggregator = tio.inference.GridAggregator(grid_sampler)

        # with torch.no_grad():
        for patches_batch in patch_loader:
            mod1_tensor = patches_batch['mod1'][tio.DATA].float()
            mod2_tensor = patches_batch['mod2'][tio.DATA].float()
            locations = patches_batch[tio.LOCATION]
            logits = self.model(mod1_tensor, mod2_tensor)
            print(f'Unique logits: {torch.unique(logits)}')
            print(f'Averaged logits: {torch.mean(logits)}')
            for i in range(len(mod1_tensor.shape) - 1):
                logits = logits.unsqueeze(-1)
            logits = logits.expand(*mod1_tensor.shape, )
            labels = (torch.sigmoid(logits) > 0.5).float()
            # labels = torch.clone(logits)
            outputs = labels
            aggregator.add_batch(outputs, locations)
        output_tensor = aggregator.get_output_tensor()
        # TODO: ask if we could also directly use the logits instead of the sigmoid
        return torch.mean(output_tensor)
