import os
import fire
from pytorch_lightning import Trainer
import torch 
from util import init_exp_folder, Args, get_ckpt_callback, get_early_stop_callback, get_logger
from model import get_task, load_task
import json 
import pickle

""" 
- Number of MLP blocks
- Size of hidden layer (MLP Ratio) 
- Dropout probability 
- Learning rate 
- Batch size 
- Trying out solely 3D joints 
"""

def train(save_dir="/workspace/experiments/simulation/mixer_results",
          exp_name="test_1",
          gpus=1, 
          pretrained=True,
          num_classes=3,
          accelerator=None,
          gradient_clip_val=0.5,
          max_epochs=100,
          patience=50,
          limit_train_batches=1.0,
          tb_path="/workspace/experiments/simulation/mixer_results/tb", 
          loss_fn="BCE",
          learn_rate=1e-4,
          batch_size=16,
          optimizer="Adam",
          dataset_path="", 
          proj_name="hmr-mixer", 
          weight_decay=0, 
          embedd_dim=381,
          seq_len=50, 
          num_mlp_blocks=8,
          mlp_ratio=(0.5, 4.0), 
          dropout_prob=0.0, 
          metrics_strategy="weighted",
          oversample=False 
          ):
    """
    Run Trainer 
    """
    args = Args(locals())
    init_exp_folder(args)
    task = get_task(args)
    wandb_hps = {"hp" : 0} # Add hyperparams 
    exp_dir_path = os.path.join(save_dir, exp_name)
    logger = get_logger(save_dir, exp_name, wandb_hps=wandb_hps, project=proj_name)
    if gpus > 1:
        accelerator='ddp'
    trainer = Trainer(devices=gpus,
                      accelerator=accelerator,
                      logger=logger,
                      callbacks=[get_early_stop_callback(patience),
                                 get_ckpt_callback(save_dir, exp_name, "ckpt")],
                      gradient_clip_val=gradient_clip_val,
                      limit_train_batches=limit_train_batches,
                      max_epochs=max_epochs,
                      )
    trainer.fit(task)
    test(exp_dir_path=exp_dir_path) 
    #TODO - include testing call right after traiing 
    return save_dir, exp_name


def test(ckpt_path=None,
         ckpt_suffix="ckpt.ckpt",
         exp_dir_path="", 
         proj_name="hmr-mixer", 
         new_model=True,
         log_exp_name="",
         save_error_analysis=True,
         save_dir="",  
         use_test_set=False,
         **kwargs):
    """
    Run the testing experiment.

    Args:
        ckpt_path: Path for the experiment to load
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
    Returns: None
    """
    if not ckpt_path: 
        ckpt_path = os.path.join(exp_dir_path, "ckpts", ckpt_suffix)
    assert os.path.exists(ckpt_path)
    args_path = os.path.join(exp_dir_path, "args.json") 
    with open(args_path) as file:
        args_dict = json.load(file)
    save_dir = args_dict['save_dir'] 
    results_path = os.path.join(save_dir, "results.pkl")
    indiv_save_path = os.path.join(save_dir, "analysis.csv")
    wandb_hps = args_dict 
    logger = get_logger(save_path=args_dict['save_dir'], exp_name=args_dict['exp_name'], test=True, 
                        wandb_hps=wandb_hps, project=proj_name, log_exp_name=log_exp_name)
    args_dict['results_path'] = results_path 
    args_dict['save_path'] = indiv_save_path 
    args_dict['save_error_analysis'] = save_error_analysis 
    trainer = Trainer(devices=1, logger=logger)
    task = load_task(ckpt_path, **args_dict) 
    trainer.test(task)

def train_wrapper(save_dir="/pasteur/u/bencliu/baseline/experiments/simulation/mixer_results",
          exp_name="test_1",
          devices=1, 
          pretrained=True,
          num_classes=3,
          accelerator=None,
          gradient_clip_val=0.5,
          max_epochs=200,
          patience=40,
          limit_train_batches=1.0,
          tb_path="/pasteur/u/bencliu/baseline/experiments/simulation/mixer_results/tb", 
          loss_fn="BCE",
          learn_rate=5e-4,
          batch_size=16,
          optimizer="Adam",
          dataset_path="", 
          proj_name="hmr-mixer", 
          weight_decay=0, 
          embedd_dim=381,
          seq_len=50, 
          num_mlp_blocks=8,
          mlp_ratio=(0.5, 4.0), 
          dropout_prob=0.0, 
          ):
    """
    Run Trainer 
    """
    args = Args(locals())
    breakpoint() 
    train(args) 

#HELPER FUNCTIONS 
def write_pickle(data_object, path):
    with open(path, 'wb') as handle:
        pickle.dump(data_object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding='latin1')

if __name__ == "__main__":
    print("Started main function")
    fire.Fire()
