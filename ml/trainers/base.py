"""
Common PyTorch trainer code.
"""

# System
import os
import time
import logging

# Externals
import numpy as np
import pandas as pd
import torch

# Locals
import utils

def _format_summary(summary):
    """Make a formatted string for logging summary info"""
    return ' '.join(f'{k} {v:.4g}' for (k, v) in summary.items())

class BaseTrainer(object):
    """
    Base class for PyTorch trainers.
    This implements the common training logic,
    logging of summaries, and checkpoints.
    """

    def __init__(self, output_dir=None, gpu=None,
                 distributed=False, rank=0, store_inference=False):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = (os.path.expandvars(output_dir)
                           if output_dir is not None else None)
        if store_inference:
            self.store_inference = {"everyNepoch": 1, "everyNsample": 1, "outdir": "inference"}
            if isinstance(store_inference, dict):
                for k, v in store_inference.items():
                    self.store_inference[k] = v 
            self.inference_dir = self.output_dir+'/'+self.store_inference['outdir']
            os.makedirs(self.inference_dir, exist_ok=True)
        else:
            self.store_inference = False

        self.gpu = gpu
        if gpu is not None:
            self.device = 'cuda:%i' % gpu
            torch.cuda.set_device(gpu)
        else:
            self.device = 'cpu'
        self.distributed = distributed
        self.rank = rank
        self.summaries = None
        self.start_epoch = 0

    def _get_summary_file(self):
        return os.path.join(self.output_dir, 'summaries_%i.csv' % self.rank)

    def save_summary(self, summary, write_file=True):
        """Save new summary information"""

        # First summary
        if self.summaries is None:
            self.summaries = pd.DataFrame([summary])

        # Append a new summary row
        else:
            self.summaries = self.summaries.append([summary], ignore_index=True)

        # Write current summaries to file (note: overwrites each time)
        if write_file and self.output_dir is not None:
            self.summaries.to_csv(self._get_summary_file(), index=False,
                                  float_format='%.6f', sep='\t')

    def load_summaries(self):
        self.summaries = pd.read_csv(self._get_summary_file(), delim_whitespace=True)

    def save_inference(self, inference, epoch):
        if not self.store_inference: return
        if epoch % self.store_inference["everyNepoch"]: return
        for batch_info, output in inference:
            batch_i, batch_f = batch_info
            if batch_i[0] % self.store_inference["everyNsample"]: continue
            np.save(self.inference_dir+'/'+"epoch%d-%s"%(epoch, batch_f[0].replace("xy", "yinf")), np.array(output))

    def _get_checkpoint_file(self, checkpoint_id):
        return os.path.join(self.output_dir, 'checkpoints',
                            'checkpoint_%03i.pth.tar' % checkpoint_id)

    def write_checkpoint(self, checkpoint_id):
        """Write a checkpoint for the trainer"""
        assert self.output_dir is not None
        checkpoint_file = self._get_checkpoint_file(checkpoint_id)
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_id=-1):
        """Load from checkpoint"""
        assert self.output_dir is not None

        # Now load the checkpoint
        try:
            rank = self.rank
            self.rank = 0
            self.load_summaries()
            if checkpoint_id == -1:
                checkpoint_id = self.summaries.epoch.iloc[-1]
            self.start_epoch = self.summaries.epoch.max() + 1
            checkpoint_file = self._get_checkpoint_file(checkpoint_id)
            self.logger.info('Loading checkpoint at %s', checkpoint_file)
            self.load_state_dict(torch.load(checkpoint_file, map_location=self.device))
        except FileNotFoundError:
            self.logger.info('No summaries 0 file found. Will not load checkoint')
        self.rank = rank
        self.summaries = None

        # load the summaries
        try:
            self.load_summaries()
        except FileNotFoundError:
            self.logger.info('No summaries file found.')

    def state_dict(self):
        """Virtual method to return state dict for checkpointing"""
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        """Virtual method to load a state dict from a checkpoint"""
        raise NotImplementedError

    def build(self, config):
        """Virtual method to build model, optimizer, etc."""
        raise NotImplementedError

    def train_epoch(self, data_loader):
        """Virtual method to train a model"""
        raise NotImplementedError

    def evaluate(self, data_loader):
        """Virtual method to evaluate a model"""
        raise NotImplementedError

    def train(self, train_data_loader, n_epochs, valid_data_loader=None):
        """Run the model training"""

        # Loop over epochs
        if train_data_loader is not None:
            for i in range(self.start_epoch, n_epochs):
                utils.distributed.try_barrier()

                self.logger.info('Epoch %i', i)
                summary = dict(epoch=i)

                # Train on this epoch
                start_time = time.time()
                train_summary = self.train_epoch(train_data_loader)
                train_summary['time'] = time.time() - start_time
                self.logger.info('Train: %s', _format_summary(train_summary))
                for (k, v) in train_summary.items():
                    summary[f'train_{k}'] = v

                # Evaluate on this epoch
                if valid_data_loader is not None:
                    start_time = time.time()
                    valid_summary, outputs = self.evaluate(valid_data_loader)
                    valid_summary['time'] = time.time() - start_time
                    self.logger.info('Valid: %s', _format_summary(valid_summary))
                    for (k, v) in valid_summary.items():
                        summary[f'valid_{k}'] = v
                    self.save_inference(outputs, i)

                # Save summary, checkpoint
                self.save_summary(summary)
                if self.output_dir is not None and self.rank==0:
                    self.write_checkpoint(checkpoint_id=i)

        # Just Evaluate 
        elif valid_data_loader is not None:
            utils.distributed.try_barrier()

            i = self.start_epoch - 1
            self.logger.info('Epoch %i', i)
            summary = dict(epoch=i)

            start_time = time.time()
            valid_summary, outputs = self.evaluate(valid_data_loader)
            valid_summary['time'] = time.time() - start_time
            self.logger.info('Valid: %s', _format_summary(valid_summary))
            for (k, v) in valid_summary.items():
                summary[f'valid_{k}'] = v
            self.save_inference(outputs, i)
            self.save_summary(summary, write_file=False)

        return self.summaries
