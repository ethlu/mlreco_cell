# Externals
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributed import all_reduce
import numpy as np

# Locals
from .basic import BasicTrainer

class SchedulePlateauTrainer(BasicTrainer):
    def build(self, config):
        super().build(config)
        self.scheduler_metric = config['trainer'].pop('metric')
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', 
                         verbose=True, **config['trainer'])
        self.n_ranks = config['n_ranks']

    def state_dict(self):
        basic_dict = super().state_dict()
        basic_dict['scheduler'] = self.scheduler.state_dict()
        return basic_dict

    def load_state_dict(self, state_dict):
        print(state_dict['scheduler'])
        self.scheduler.load_state_dict(state_dict.pop('scheduler'))
        super().load_state_dict(state_dict)

    def schedule(self, eval_out):
        metric_score = torch.tensor(eval_out[self.scheduler_metric])
        if self.distributed:
            all_reduce(metric_score)
        self.scheduler.step(metric_score/self.n_ranks)

def get_trainer(**kwargs):
    return SchedulePlateauTrainer(**kwargs)

