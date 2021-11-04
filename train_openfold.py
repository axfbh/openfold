import argparse
import logging
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#os.environ["MASTER_ADDR"]="10.119.81.14"
#os.environ["MASTER_PORT"]="42069"
#os.environ["NODE_RANK"]="0"

import random
import time

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin
from pytorch_lightning.plugins.environments import SLURMEnvironment
import torch

from openfold.config import model_config
from openfold.data.data_modules import (
    OpenFoldDataModule,
    #DummyDataLoader,
)
from openfold.model.model import AlphaFold
from openfold.utils.callbacks import (
    EarlyStoppingVerbose,
)
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.argparse import remove_arguments
from openfold.utils.loss import AlphaFoldLoss
from openfold.utils.seed import seed_everything
from openfold.utils.tensor_utils import tensor_tree_map
from scripts.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint
)

from openfold.utils.logger import PerformanceLoggingCallback


class OpenFoldWrapper(pl.LightningModule):
    def __init__(self, config):
        super(OpenFoldWrapper, self).__init__()
        self.config = config
        self.model = AlphaFold(config)
        self.loss = AlphaFoldLoss(config.loss)
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        if(self.ema.device != batch["aatype"].device):
            self.ema.to(batch["aatype"].device)

        # Run the model
        outputs = self(batch)
        
        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Compute loss
        loss = self.loss(outputs, batch)

        #if(torch.isnan(loss) or torch.isinf(loss)):
        #    logging.warning("loss is NaN. Skipping example...")
        #    loss = loss.new_tensor(0., requires_grad=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if(self.cached_weights is None):
            self.cached_weights = model.state_dict()
            self.model.load_state_dict(self.ema.state_dict()["params"])
        
        # Calculate validation loss
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)
        loss = self.loss(outputs, batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, _):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def configure_optimizers(self, 
        learning_rate: float = 1e-3,
        eps: float = 1e-8
    ) -> torch.optim.Adam:
        # Ignored as long as a DeepSpeed optimizer is configured
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            eps=eps
        )

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()


def main(args):
    if(args.seed is not None):
        seed_everything(args.seed) 

    config = model_config( # fixes num_workers at 1, increase to number of CPUs?
        "model_1", 
        train=True, 
        low_prec=(args.precision == 16)
    ) 
    model_module = OpenFoldWrapper(config)
    if(args.resume_from_ckpt and args.resume_model_weights_only):
        sd = get_fp32_state_dict_from_zero_checkpoint(args.resume_from_ckpt)
        sd = {k[len("module."):]:v for k,v in sd.items()}
        model_module.load_state_dict(sd)
        logging.info("Successfully loaded model weights...")
    #data_module = DummyDataLoader("batch.pickle")
    data_module = OpenFoldDataModule(
        config=config.data, 
        batch_seed=args.seed,
        **vars(args)
    )
    data_module.prepare_data()
    data_module.setup()

    callbacks = []
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if(args.checkpoint_best_val):
        mc = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="openfold_{epoch}_{step}_{val_loss:.2f}",
            monitor="val_loss",
        )
    else:
        mc = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="openfold_{epoch}_{step}"
        )
    callbacks.append(mc)

    if(args.early_stopping):
        es = EarlyStoppingVerbose(
            monitor="val_loss",
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode="min",
            check_finite=True,
            strict=True,
        )
        callbacks.append(es)
    if args.log_performance:
        global_batch_size = args.num_nodes * args.gpus
        perf = PerformanceLoggingCallback(
            log_file=os.path.join(args.output_dir, "performance_log.json"),
            global_batch_size=global_batch_size,
        )
        callbacks.append(perf)

    if(args.deepspeed_config_path is not None):
        if "SLURM_JOB_ID" in os.environ:
            cluster_environment = SLURMEnvironment()
        else:
            cluster_environment = None
        strategy = DeepSpeedPlugin(
            config=args.deepspeed_config_path,
            cluster_environment=cluster_environment,
        )
    elif args.gpus > 1 or args.num_nodes > 1:
        strategy = "ddp"
    else:
        strategy = None
    
    #if args.resume_from_checkpoint is not None:
    #    trainer = pl.Trainer(gpus = args.gpus, resume_from_checkpoint = args.resume_from_checkpoint)
    #else:
    trainer = pl.Trainer.from_argparse_args(
        args,
        #plugins=plugins,
        #callbacks=callbacks
        strategy=strategy,
        callbacks=callbacks,
    )

    if(args.resume_model_weights_only):
        ckpt_path = None
    else:
        ckpt_path = args.resume_from_ckpt
    t0 = time.time()
    trainer.fit(
        model_module, 
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )
    t1 = time.time()
    print(f'Fitting time per epoch = {(t1 - t0) / args.max_epochs}')
    #print(f'TRAINER={trainer}\n')
    #if args.resume_from_checkpoint is None:
    #    trainer.fit(model_module, datamodule=data_module)
    #else:
    #    trainer.fit(model_module, datamodule=data_module)#, ckpt_path=args.resume_from_checkpoint)
    #trainer.save_checkpoint("final.ckpt")
    trainer.save_checkpoint(
        os.path.join(trainer.logger.log_dir, "checkpoints", "final.ckpt")
    )


def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_dir", type=str,
        help="Directory containing training mmCIF files"
    )
    parser.add_argument(
        "train_alignment_dir", type=str,
        help="Directory containing precomputed training alignments"
    )
    parser.add_argument(
        "template_mmcif_dir", type=str,
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "output_dir", type=str,
        help='''Directory in which to output checkpoints, logs, etc. Ignored
                if not on rank 0'''
    )
    parser.add_argument(
        "max_template_date", type=str,
        help='''Cutoff for all templates. In training mode, templates are also 
                filtered by the release date of the target'''
    )
    parser.add_argument(
        "--distillation_data_dir", type=str, default=None,
        help="Directory containing training PDB files"
    )
    parser.add_argument(
        "--distillation_alignment_dir", type=str, default=None,
        help="Directory containing precomputed distillation alignments"
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation mmCIF files"
    )
    parser.add_argument(
        "--val_alignment_dir", type=str, default=None,
        help="Directory containing precomputed validation alignments"
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default='/usr/bin/kalign',
        help="Path to the kalign binary"
    )
    parser.add_argument(
        "--train_mapping_path", type=str, default=None,
        help='''Optional path to a .json file containing a mapping from
                consecutive numerical indices to sample names. Used to filter
                the training set'''
    )
    parser.add_argument(
        "--distillation_mapping_path", type=str, default=None,
        help="""See --train_mapping_path"""
    )
    parser.add_argument(
        "--template_release_dates_cache_path", type=str, default=None,
        help="""Output of scripts/generate_mmcif_cache.py run on template mmCIF
                files."""
    )
    parser.add_argument(
        "--use_small_bfd", type=bool_type, default=False,
        help="Whether to use a reduced version of the BFD database"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=None,
        help="Path to DeepSpeed config. If not provided, DeepSpeed is disabled"
    )
    parser.add_argument(
        "--checkpoint_best_val", type=bool_type, default=True,
        help="""Whether to save the model parameters that perform best during
                validation"""
    )
    parser.add_argument(
        "--early_stopping", type=bool_type, default=False,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0,
        help="""The smallest decrease in validation loss that counts as an 
                improvement for the purposes of early stopping"""
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--log_performance", action='store_true',
        help="Measure performance"
    )
    parser = pl.Trainer.add_argparse_args(parser)
   
    # Disable the initial validation pass
    parser.set_defaults(
        num_sanity_val_steps=0,
    )

    # Remove some buggy/redundant arguments introduced by the Trainer
    remove_arguments(parser, ["--accelerator", "--resume_from_checkpoint"]) 

    args = parser.parse_args()
    print(f'train_openfold.py: {args}')
    if(args.seed is None and 
        ((args.gpus is not None and args.gpus > 1) or 
         (args.num_nodes is not None and args.num_nodes > 1))):
        raise ValueError("For distributed training, --seed must be specified")

    main(args)
#(openfold_venv) u00u98too4mkqFBu8M357@dgx1-003:~/openfold$ python3 train_openfold.py mmcif_dir_new/ alignment_dir_new/ template_mmcif_dir/ train_op_new/    2021-10-10 --template_release_dates_cache_path mmcif_cache_new.json --precision 16 --gpus 8 --replace_sampler_ddp=True --accelerator ddp --seed 42 --deepspeed_config_path deepspeed_config.json --default_root_dir train_op_new  --resume_from_checkpoint 'train_op/lightning_logs/version_0/checkpoints/epoch=10-step=10.ckpt'

#train_openfold.py: Namespace(accumulate_grad_batches=None, amp_backend='native', amp_level=None, auto_lr_find=False, auto_scale_batch_size=False, auto_select_gpus=False, benchmark=False, check_val_every_n_epoch=1, checkpoint_best_val=True, checkpoint_callback=None, deepspeed_config_path='deepspeed_config.json', default_root_dir='train_op_8', detect_anomaly=False, deterministic=False, devices=None, distillation_alignment_dir=None, distillation_data_dir=None, distillation_mapping_path=None, early_stopping=False, enable_checkpointing=True, enable_model_summary=True, enable_progress_bar=True, fast_dev_run=False, flush_logs_every_n_steps=None, gpus=8, gradient_clip_algorithm=None, gradient_clip_val=None, ipus=None, kalign_binary_path='/usr/bin/kalign', limit_predict_batches=1.0, limit_test_batches=1.0, limit_train_batches=1.0, limit_val_batches=1.0, log_every_n_steps=50, log_gpu_memory=None, logger=True, max_epochs=None, max_steps=-1, max_template_date='2021-10-10', max_time=None, min_delta=0, min_epochs=None, min_steps=None, move_metrics_to_cpu=False, multiple_trainloader_mode='max_size_cycle', num_nodes=1, num_processes=1, num_sanity_val_steps=0, output_dir='train_op_8', overfit_batches=0.0, patience=3, plugins=None, precision=16, prepare_data_per_node=None, process_position=0, profiler=None, progress_bar_refresh_rate=None, reload_dataloaders_every_epoch=False, reload_dataloaders_every_n_epochs=0, replace_sampler_ddp=True, resume_from_ckpt=None, resume_model_weights_only=False, seed=42, stochastic_weight_avg=False, strategy=None, sync_batchnorm=False, template_mmcif_dir='template_mmcif_dir', template_release_dates_cache_path='mmcif_cache_8.json', terminate_on_nan=None, tpu_cores=None, track_grad_norm=-1, train_alignment_dir='alignment_dir_8', train_data_dir='mmcif_dir_8', train_mapping_path=None, use_small_bfd=False, val_alignment_dir=None, val_check_interval=1.0, val_data_dir=None, weights_save_path=None, weights_summary='top')