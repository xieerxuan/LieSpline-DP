from typing import Dict

from liespline_dp.policy.base_policy import BasePolicy


class BaseLowdimRunner:
    supports_step_rollout_context = False

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy, evaluation_context=None) -> Dict:
        raise NotImplementedError()
