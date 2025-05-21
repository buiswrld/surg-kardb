import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from metrics import get_metrics_multiclass, get_metrics
from dataset import MixerDataset
import os 
import pandas as pd
import pickle
from net import MlpMixer
import torch.nn as nn
from dataset import MixerDataset


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

    def forward(self, x):
        return self.model(x.to(torch.float32)) #Necessary to prevent conversion issues ) 

    def training_step(self, batch, batch_nb):
        """
        Returns:
            A dictionary of loss and metrics, with:
                loss(required): loss used to calculate the gradient
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        x, y = batch["embedding_seq"], batch["label"]
        logits = self.forward(x) 
        loss = self.loss(logits, y.long())
        self.log("train_loss", loss)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_nb):
        x, y = batch["embedding_seq"], batch["label"].to(torch.float32)
        logits = self.forward(x) 
        loss = self.loss(logits, y.long())
        probs = torch.softmax(logits, dim=1)
        breakpoint() 
        return {'labels': y, 'logits': logits, 'probs': probs, 'val_loss': loss}

    def validation_epoch_end(self, outputs):
        """
        Aggregate and return the validation metrics
        Args:
        outputs: A list of dictionaries of metrics from `validation_step()'
        Returns: None
        Returns:
            A dictionary of loss and metrics, with:
                val_loss (required): validation_loss
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        print('validation epoch end')
        avg_loss = torch.stack([batch['val_loss'] for batch in outputs]).mean()
        labels = torch.cat([batch['labels'] for batch in outputs])
        probs = torch.cat([batch['probs'] for batch in outputs])
        metrics_strategy = self.hparams['metrics_strategy']

        #Log Val Accuracy and Loss
        self.log("val_loss", avg_loss.item())

        #Log Val Metrics
        if self.num_classes == 2:
            metrics = get_metrics(labels, probs)
        else:
            metrics = get_metrics_multiclass(labels, probs, metrics_strategy) 
        for metric_name, metric_value in metrics.items():
            self.log(f'val_{metric_name}', metric_value)

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb) 

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([batch['val_loss'] for batch in outputs]).mean()
        labels = torch.cat([batch['labels'] for batch in outputs])
        logits = torch.cat([batch['logits'] for batch in outputs])
        probs = torch.cat([batch['probs'] for batch in outputs])
        metrics_strategy = self.hparams['metrics_strategy']

        #Log Test Accuracy and Loss
        self.log("test_loss", avg_loss)

        #Log Test Metrics
        if self.num_classes == 2:
            metrics = get_metrics(labels, probs)
        else:
            metrics = get_metrics_multiclass(labels, probs, metrics_strategy) 
        for metric_name, metric_value in metrics.items():
            self.log(f'test_{metric_name}', metric_value)
        metrics['default'] = metrics['auprc']

        return {'avg_test_loss': avg_loss}

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
        dataset = MixerDataset(dataset_path, 'train')  

        #Create oversampling weights 
        if oversample:
            ref_dataset = read_pickle(dataset_path)['train']
            counts = {} 
            for example in ref_dataset: 
                label = str(example[1])
                if label not in counts:
                    counts[label] = 1 
                else:
                    counts[label] += 1 
            weights = [] 
            for example in ref_dataset: 
                label = str(example[1])
                weight = 1 / counts[label] 
                weights.append(weight) 
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights)
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        return DataLoader(dataset, shuffle=shuffle, batch_size=self.hparams['batch_size'], num_workers=8, sampler=sampler)

    def val_dataloader(self):
        dataset_path = self.hparams.get('dataset_path', "")
        dataset = MixerDataset(dataset_path, 'valid') 
        return DataLoader(dataset, shuffle=False, batch_size=self.hparams['batch_size'], num_workers=8)

    def test_dataloader(self):
        dataset_path = self.hparams.get('dataset_path', "")
        dataset = MixerDataset(dataset_path, 'test') 
        return DataLoader(dataset, shuffle=False, batch_size=self.hparams['batch_size'], num_workers=8)

#HELPER FUNCTIONS 
def write_pickle(data_object, path):
    with open(path, 'wb') as handle:
        pickle.dump(data_object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding='latin1')