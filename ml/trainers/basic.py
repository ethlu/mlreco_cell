"""
This module defines a generic trainer for simple models and datasets.
"""

# Externals
import torch
from torch.nn.parallel import DistributedDataParallel

# Locals
from .base import BaseTrainer
from models import get_model
import utils.metrics

class BasicTrainer(BaseTrainer):
    """Trainer code for basic single-model problems"""

    def __init__(self, **kwargs):
        super(BasicTrainer, self).__init__(**kwargs)

    def build(self, config):
        """Instantiate our model, optimizer, loss function"""

        # Construct the model
        self.model = get_model(**config['model']).to(self.device)
        if self.distributed:
            device_ids = [self.gpu] if self.gpu is not None else None
            self.model = DistributedDataParallel(self.model, device_ids=device_ids)

        # Construct the loss function
        loss_config = config['loss']
        Loss = getattr(torch.nn, loss_config.pop('name'))
        self.loss_func = Loss(**loss_config)

        # Construct the optimizer
        optimizer_config = config['optimizer']
        Optim = getattr(torch.optim, optimizer_config.pop('name'))
        self.optimizer = Optim(self.model.parameters(), **optimizer_config)

        # Construct the metrics
        metrics_config = config.get('metrics', {})
        self.metrics = utils.metrics.get_metrics(metrics_config)

        # Print a model summary
        if self.rank == 0:
            self.logger.info(self.model)
            self.logger.info('Number of parameters: %i',
                             sum(p.numel() for p in self.model.parameters()))
    
    def state_dict(self):
        """Trainer state dict for checkpointing"""
        return dict(
            model=(self.model.module.state_dict()
                   if self.distributed
                   else self.model.state_dict()),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        if self.distributed:
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def train_epoch(self, data_loader):
        """Train for one epoch"""

        self.model.train()

        # Reset metrics
        sum_loss = 0
        utils.metrics.reset_metrics(self.metrics)

        # Loop over training batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            self.model.zero_grad()
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target)
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()
            utils.metrics.update_metrics(self.metrics, batch_output, batch_target)
            self.logger.debug('batch %i loss %.3f', i, batch_loss.item())

        train_loss = sum_loss / (i + 1)
        metrics_summary = utils.metrics.get_results(self.metrics)
        self.logger.debug('Processed %i batches' % (i + 1))

        # Return summary
        return dict(loss=train_loss, **metrics_summary)

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""

        self.model.eval()

        # Reset metrics
        sum_loss = 0
        utils.metrics.reset_metrics(self.metrics)

        # Loop over batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target).item()
            sum_loss += batch_loss
            utils.metrics.update_metrics(self.metrics, batch_output, batch_target)
            self.logger.debug('batch %i loss %.3f', i, batch_loss)

        # Summarize validation metrics
        metrics_summary = utils.metrics.get_results(self.metrics)

        valid_loss = sum_loss / (i + 1)
        self.logger.debug('Processed %i samples in %i batches',
                          len(data_loader.sampler), i + 1)

        # Return summary
        return dict(loss=valid_loss, **metrics_summary)

def get_trainer(**kwargs):
    return BasicTrainer(**kwargs)

def _test():
    t = BasicTrainer(output_dir='./')
    t.build_model()
