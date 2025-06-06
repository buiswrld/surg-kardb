import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
import numpy as np
from metrics import get_metrics_multiclass, get_metrics
from dataset import MixerDataset
import os 
import pandas as pd
import pickle
from net import MlpMixer
import torch.nn as nn
from dataset import MixerDataset
from gcn_dataset import GNNDataset


def get_task(args):
    return MixerTask(args)

def load_task(ckpt_path, **kwargs):
    task = MixerTask(kwargs)
    return task.load_from_checkpoint(ckpt_path, **kwargs)

class MixerTask(pl.LightningModule):
    """Standard interface for the trainer to interact with the model."""
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        hmr_embedd_dim = self.hparams.get('embedd_dim', 0) 
        seq_len = self.hparams.get('seq_len', 145) 
        num_mlp_blocks = self.hparams.get('num_mlp_blocks', 8) 
        mlp_ratio = self.hparams.get('mlp_ratio', (0.5, 4.0)) 
        dropout_prob = self.hparams.get('dropout_prob', 0.0) 
        num_classes = self.hparams.get('num_classes', 2) 
        self.num_classes = num_classes
        self.model = MlpMixer(hmr_embedd_dim=hmr_embedd_dim, seq_len=seq_len, num_classes=num_classes,
                              drop_path_rate=dropout_prob, mlp_ratio=mlp_ratio, num_blocks=num_mlp_blocks)
        self.loss = nn.CrossEntropyLoss()
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, x):
        return self.model(x.to(torch.float32))

    def training_step(self, batch, batch_nb):
        x, y = batch.x, batch.y
        logits = self.forward(x) 
        loss = self.loss(logits, y.long())
        self.log("train_loss", loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch.x, batch.y
        logits = self.forward(x) 
        loss = self.loss(logits, y.long())
        probs = torch.softmax(logits, dim=1)
        self.validation_outputs.append({
            'labels': y, 'logits': logits, 'probs': probs, 'val_loss': loss
        })
        return {'loss': loss}

    def on_validation_epoch_end(self):
        print('validation epoch end')
        avg_loss = torch.stack([out['val_loss'] for out in self.validation_outputs]).mean()
        labels = torch.cat([out['labels'] for out in self.validation_outputs])
        probs = torch.cat([out['probs'] for out in self.validation_outputs])
        metrics_strategy = self.hparams['metrics_strategy']

        self.log("val_loss", avg_loss.item())

        if self.num_classes == 2:
            metrics = get_metrics(labels, probs)
        else:
            metrics = get_metrics_multiclass(labels, probs, metrics_strategy)

        for metric_name, metric_value in metrics.items():
            self.log(f'val_{metric_name}', metric_value)

        self.validation_outputs.clear()

    def test_step(self, batch, batch_nb):
        x, y = batch.x, batch.y
        logits = self.forward(x) 
        loss = self.loss(logits, y.long())
        probs = torch.softmax(logits, dim=1)
        self.test_outputs.append({
            'labels': y, 'logits': logits, 'probs': probs, 'val_loss': loss
        })
        return {'loss': loss}

    def on_test_epoch_end(self):
        avg_loss = torch.stack([out['val_loss'] for out in self.test_outputs]).mean()
        labels = torch.cat([out['labels'] for out in self.test_outputs])
        logits = torch.cat([out['logits'] for out in self.test_outputs])
        probs = torch.cat([out['probs'] for out in self.test_outputs])
        metrics_strategy = self.hparams['metrics_strategy']

        self.log("test_loss", avg_loss)

        if self.num_classes == 2:
            metrics = get_metrics(labels, probs)
        else:
            metrics = get_metrics_multiclass(labels, probs, metrics_strategy)

        for metric_name, metric_value in metrics.items():
            self.log(f'test_{metric_name}', metric_value)

        self.test_outputs.clear()

    def configure_optimizers(self):
        learn_rate = self.hparams['learn_rate']
        if self.hparams['optimizer'] == 'Adam':
            weight_decay = self.hparams.get('weight_decay', 0)  
            return [torch.optim.Adam(self.parameters(), lr=learn_rate, weight_decay=weight_decay)]
        elif self.hparams['optimizer'] == 'AdamW':
            weight_decay = self.hparams.get('weight_decay', 1e-5) 
            return [torch.optim.AdamW(self.parameters(), lr=learn_rate, weight_decay=weight_decay)]
        else:
            return [torch.optim.SGD(self.parameters(), lr=learn_rate, momentum=0.9)]

    def train_dataloader(self):
        oversample = self.hparams['oversample']
        dataset_path = self.hparams.get('dataset_path', "")
        dataset = GNNDataset(
            pkl_path=self.hparams['dataset_path'],
            split="train",
            seq_len=self.hparams.get('seq_len', 5),
            num_joints=self.hparams.get('num_joints', 28),
            coords_per_joint=self.hparams.get('coords_per_joint', 3)
        )

        if oversample:
            ref_dataset = read_pickle(dataset_path)['train']
            counts = {} 
            for example in ref_dataset: 
                label = str(example[1])
                counts[label] = counts.get(label, 0) + 1
            weights = [1 / counts[str(example[1])] for example in ref_dataset]
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
            shuffle = False
        else:
            sampler = None
            shuffle = True
        return DataLoader(dataset, shuffle=shuffle, batch_size=self.hparams['batch_size'], num_workers=8, sampler=sampler)

    def val_dataloader(self):
        dataset_path = self.hparams.get('dataset_path', "")
        dataset = GNNDataset(
            pkl_path=self.hparams['dataset_path'],
            split="valid",
            seq_len=self.hparams.get('seq_len', 5),
            num_joints=self.hparams.get('num_joints', 28),
            coords_per_joint=self.hparams.get('coords_per_joint', 3)
        )
        return DataLoader(dataset, shuffle=False, batch_size=self.hparams['batch_size'], num_workers=8)

    def test_dataloader(self):
        dataset_path = self.hparams.get('dataset_path', "")
        dataset = GNNDataset(
            pkl_path=self.hparams['dataset_path'],
            split="test",
            seq_len=self.hparams.get('seq_len', 5),
            num_joints=self.hparams.get('num_joints', 28),
            coords_per_joint=self.hparams.get('coords_per_joint', 3)
        )
        return DataLoader(dataset, shuffle=False, batch_size=self.hparams['batch_size'], num_workers=8)

#HELPER FUNCTIONS 
def write_pickle(data_object, path):
    with open(path, 'wb') as handle:
        pickle.dump(data_object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding='latin1')
