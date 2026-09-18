from typing import Dict

import torch

from liespline_dp.model.common.module_attr_mixin import ModuleAttrMixin
from liespline_dp.model.common.normalizer import LinearNormalizer


class BasePolicy(ModuleAttrMixin):
    """Common inference interface for all observation modalities and algorithms."""

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
