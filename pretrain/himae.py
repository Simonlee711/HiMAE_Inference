import pandas as pd
import os
import s3fs
import h5py
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import warnings
import gc
import matplotlib.pyplot as plt
import json
from io import StringIO
from tqdm import tqdm
import time # Added for runtime calculation

# Added for FLOPs calculation
try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import LambdaLR
import wandb
import boto3

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
warnings.filterwarnings("ignore")

CONSOLIDATED_METADATA_PATH = '' # PRETRAINING DATASOURCE

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

if FlopCountAnalysis is None:
    logger.warning("`fvcore` not installed. FLOPs calculation will be skipped. Install with 'pip install fvcore'")


def set_seed(seed=42):
    L.seed_everything(seed)


class BasePPGDataset(Dataset):
    def __init__(self, in_memory_data, f_s, T, C):
        self.in_memory_data = in_memory_data
        self.f_s = f_s
        self.T = T
        self.L = f_s * T
        self.C = C

    def __len__(self):
        return len(self.in_memory_data)

class PPGOnlyDataset(BasePPGDataset):
    def __init__(self, in_memory_data, f_s, T):
        super().__init__(in_memory_data, f_s, T, C=1)

    def __getitem__(self, idx):
        try:
            original_signal = self.in_memory_data[idx]
            
            signal = np.nan_to_num(original_signal.astype(np.float32))
            v_min = signal.min()
            v_max = signal.max()

            if (v_max - v_min) > 1e-6:
                scaled_signal = -1.0 + 2.0 * (signal - v_min) / (v_max - v_min)
            else:
                scaled_signal = np.zeros_like(signal)

            processed_signal = np.pad(scaled_signal, (0, self.L - len(scaled_signal)), 'edge')[:self.L]
            X = processed_signal.reshape(self.L, 1).astype(np.float32)

            if X.shape[1] != self.C:
                raise ValueError(f"Channel mismatch! Expected {self.C}, got {X.shape[1]}")
            return torch.from_numpy(X)
        except Exception as e:
            logger.warning(f"Error at index {idx} in in-memory data: {e}. Returning zeros.")
            return torch.zeros(self.L, self.C, dtype=torch.float32)

class PPGOnlyDataModule(L.LightningDataModule):
    def __init__(self, consolidated_meta_path, f_s, T, batch_size, num_workers, train_frac=0.8, test_frac=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.C = 1
        self.in_memory_data = None
        
    def prepare_data(self):
        logger.info("Preloading all data into RAM...")
        
        try:
            meta_df = pd.read_csv(self.hparams.consolidated_meta_path)
        except Exception as e:
            logger.error(f"Failed to read metadata from local path: {e}")
            raise e
        
        all_signals = []
        for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Loading HDF5 data into RAM"):
            shard_path = row['local_path']
            global_idx = row['global_idx']

            try:
                with h5py.File(shard_path, 'r') as h5_file:
                    original_signal = h5_file[global_idx]['normalized_waveform'][:]
                    all_signals.append(original_signal)
            except Exception as e:
                logger.warning(f"Skipping sample {global_idx} from {shard_path} due to error: {e}")
        
        self.in_memory_data = np.stack(all_signals, axis=0)
        logger.info(f"All data loaded into RAM. Shape: {self.in_memory_data.shape}, Dtype: {self.in_memory_data.dtype}")

    def setup(self, stage=None):
        full_dataset = PPGOnlyDataset(in_memory_data=self.in_memory_data, f_s=self.hparams.f_s, T=self.hparams.T)
        n_total = len(full_dataset)
        n_train = int(n_total * self.hparams.train_frac)
        n_test = int(n_total * self.hparams.test_frac)
        n_val = n_total - n_train - n_test
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
        )
        logger.info(f"Data split: Train {len(self.train_dataset)}, Val {len(self.val_dataset)}, Test {len(self.test_dataset)}.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=True)


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        padding = 2
        self.main_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        self.final_activation = nn.GELU()

    def forward(self, x):
        main_out = self.main_path(x)
        shortcut_out = self.shortcut(x)
        if main_out.size() != shortcut_out.size():
            target_size = main_out.size()[2]
            shortcut_out = shortcut_out[:, :, :target_size]
        return self.final_activation(main_out + shortcut_out)


class DecoderSkipBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv_block = nn.Sequential(
            nn.Conv1d(out_channels + skip_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        if x.size() != skip_connection.size():
            target_size = skip_connection.size()[2]
            x = x[:, :, :target_size]
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv_block(x)


class HiMAE(nn.Module):
    def __init__(self, in_chans=1, seq_len=750, channels=[16, 32, 64, 128]):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        current_chans = in_chans
        for chan in channels:
            self.encoder_layers.append(ResConvBlock(current_chans, chan, stride=2))
            current_chans = chan

        reversed_channels = channels[::-1]
        for i in range(len(reversed_channels) - 1):
            in_c, skip_c, out_c = reversed_channels[i], reversed_channels[i+1], reversed_channels[i+1]
            self.decoder_layers.append(DecoderSkipBlock(in_c, skip_c, out_c))

        self.final_deconv = nn.ConvTranspose1d(channels[0], in_chans, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.final_activation = nn.Tanh()

    def forward(self, x):
        skip_connections = []
        current_x = x
        for encoder_block in self.encoder_layers:
            current_x = encoder_block(current_x)
            skip_connections.append(current_x)

        bottleneck = skip_connections.pop()
        skip_connections = skip_connections[::-1]

        current_x = bottleneck
        for i, decoder_block in enumerate(self.decoder_layers):
            skip = skip_connections[i]
            current_x = decoder_block(current_x, skip)

        x_reconstructed = self.final_deconv(current_x)

        if x_reconstructed.shape[-1] != self.seq_len:
            x_reconstructed = nn.functional.interpolate(x_reconstructed, size=self.seq_len, mode='linear', align_corners=False)

        return self.final_activation(x_reconstructed)


class HiMAELightningModule(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = HiMAE(
            in_chans=self.hparams.C,
            seq_len=self.hparams.T * self.hparams.f_s,
            channels=self.hparams.channels
        )
        self.patch_size = self.hparams.patch
        self.seq_len = self.hparams.T * self.hparams.f_s
        self.criterion = nn.MSELoss(reduction='none')

    def random_masking(self, x):
        B, C, L = x.shape
        num_patches = L // self.patch_size
        if L % self.patch_size != 0:
            raise ValueError("Sequence length must be divisible by patch size.")

        num_masked_patches = int(self.hparams.mask_ratio * num_patches)
        noise = torch.rand(B, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_mask = ids_shuffle[:, :num_masked_patches]

        patch_mask = torch.zeros(B, num_patches, device=x.device)
        patch_mask.scatter_(1, ids_mask, 1.0)

        seq_mask = patch_mask.repeat_interleave(self.patch_size, dim=1).unsqueeze(1)
        return seq_mask

    def _common_step(self, batch, stage):
        x = batch.transpose(1, 2)
        mask = self.random_masking(x)
        x_masked_input = x * (1 - mask)
        reconstruction = self.model(x_masked_input)
        loss = (((reconstruction - x)**2) * mask).sum() / (mask.sum() + 1e-9)
        
        reconstructed_signal_with_original_unmasked_parts = reconstruction * mask + x * (1 - mask)

        
        self.log(f'{stage}/mse_loss', loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx): return self._common_step(batch, 'train')
    def validation_step(self, batch, batch_idx): return self._common_step(batch, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        try:
            total_steps = len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
        except Exception:
            logger.warning("Could not calculate total steps. Using fallback.")
            total_steps = self.hparams.pretrain_epochs * 250
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        scheduler = {'scheduler': LambdaLR(optimizer, lr_lambda), 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]


class HiMAEReconstructionVisualizer(L.Callback):
    def __init__(self, dataloader, num_samples=4, title_suffix="", every_n_epochs=5):
        super().__init__()
        self.dataloader = dataloader
        self.num_samples = num_samples
        self.title_suffix = title_suffix
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        if not trainer.is_global_zero: return
        pl_module.eval()
        batch = next(iter(self.dataloader)).to(pl_module.device)
        with torch.no_grad():
            x = batch.transpose(1, 2)
            mask = pl_module.random_masking(x)
            x_masked_input = x * (1 - mask)
            reconstruction = pl_module.model(x_masked_input)
            
            reconstructed_full_signal = reconstruction * mask + x * (1 - mask)
            
        plots_to_log = {}
        num_to_plot = min(self.num_samples, batch.shape[0])
        for i in range(num_to_plot):
            original_signal = x[i].squeeze().cpu().numpy()
            reconstructed_signal = reconstructed_full_signal[i].squeeze().cpu().numpy()
            mask_vis = mask[i].squeeze().cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(18, 6))
            ax.plot(original_signal, label='Original Signal', color='blue', alpha=0.7)
            
            reconstructed_only_masked = np.where(mask_vis == 1, reconstructed_signal, np.nan)
            ax.plot(reconstructed_only_masked, label='Reconstructed Masked Region', color='green', linestyle=':', linewidth=2)
            
            mask_indices = np.where(mask_vis == 1)[0]
            if mask_indices.size > 0:
                start = mask_indices[0]
                end = start
                for idx in mask_indices[1:]:
                    if idx == end + 1:
                        end = idx
                    else:
                        ax.axvspan(start, end + 1, color='red', alpha=0.3)
                        start = idx
                        end = idx
                ax.axvspan(start, end + 1, color='red', alpha=0.3, label='Masked Region (Input)' if 'Masked Region (Input)' not in [l.get_label() for l in ax.legend().get_lines()] else "")

            ax.set_title(f'Sample {i} Reconstruction {self.title_suffix} (Epoch {trainer.current_epoch})')
            ax.set_xlabel('Time Steps'); ax.set_ylabel('Amplitude')
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plots_to_log[f"Reconstructions/Sample_{i}_Epoch_{trainer.current_epoch}"] = wandb.Image(fig)
            plt.close(fig)
        trainer.logger.experiment.log(plots_to_log, commit=False)
        pl_module.train()

# --- New Callback for timing epochs ---
class EpochTimer(L.Callback):
    """Callback to time each training epoch and log the duration."""
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.start_time = 0

    def on_train_epoch_start(self, trainer, pl_module):
        """Record the start time at the beginning of each training epoch."""
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        """Calculate and log the epoch duration at the end of each training epoch."""
        end_time = time.time()
        duration = end_time - self.start_time
        self.epoch_times.append(duration)
        logger.info(f"Epoch {trainer.current_epoch + 1} duration: {duration:.2f} seconds")


def main_HiMAE():
    logger.info("--- Starting HiMAE Model Training ---")
    set_seed(42)

    cnn_args = dict(
        consolidated_meta_path=CONSOLIDATED_METADATA_PATH,
        f_s=100, T=10, patch=5, C=1,
        channels=[16, 32, 64, 128, 256],
        mask_ratio=0.8, batch_size=1024, pretrain_epochs=300, patience=3,
        lr=1e-3, weight_decay=1e-3,
        num_workers=max(1, os.cpu_count() // 8), train_frac=0.8, test_frac=0.1,
        warmup_ratio=0.1
    )

    data_module = PPGOnlyDataModule(
        consolidated_meta_path=cnn_args['consolidated_meta_path'],
        f_s=cnn_args['f_s'], T=cnn_args['T'],
        batch_size=cnn_args['batch_size'], num_workers=cnn_args['num_workers'],
        train_frac=cnn_args['train_frac'], test_frac=cnn_args['test_frac']
    )

    model = HiMAELightningModule(hparams=cnn_args)

    # --- FLOPs Calculation ---
    total_flops = 0
    flops_analyzer = None
    if FlopCountAnalysis:
        try:
            # Create a dummy input with the correct shape for the model (B, C, L)
            seq_len = cnn_args['f_s'] * cnn_args['T']
            C = cnn_args['C']
            dummy_input = torch.randn(1, C, seq_len)
            
            # Analyze the nn.Module part of the LightningModule
            flops_analyzer = FlopCountAnalysis(model.model, dummy_input)
            total_flops = flops_analyzer.total()

        except Exception as e:
            logger.error(f"Could not calculate FLOPs: {e}")
    # --- End FLOPs Calculation ---


    wandb_logger = WandbLogger(project="HIMAE_test", log_model="all")

    data_module.prepare_data()
    data_module.setup()

    visualizer_callback = HiMAEReconstructionVisualizer(
        data_module.val_dataloader(),
        num_samples=1,
        title_suffix="(HIMAE)",
        every_n_epochs=1
    )
    
    # --- Instantiate the new epoch timer callback ---
    epoch_timer_callback = EpochTimer()

    trainer = L.Trainer(
        max_epochs=cnn_args['pretrain_epochs'],
        accelerator='auto', devices='auto', strategy='auto',
        log_every_n_steps=20,
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(monitor='val/mse_loss', mode='min', save_top_k=-1, filename='best-cnn-mae-model'),
            EarlyStopping(monitor='val/mse_loss', patience=cnn_args['patience'], mode='min'),
            visualizer_callback,
            epoch_timer_callback # Add the timer callback to the trainer
        ]
    )

    training_duration = 0
    start_time = time.time() # Record start time
    
    try:
        logger.info("Starting HIMAE pre-training...")
        trainer.fit(model, datamodule=data_module)
        logger.info("HIMAE pre-training complete.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")

    finally:
        end_time = time.time() # Record end time
        training_duration = end_time - start_time
        
        # --- Runtime and FLOPs Reporting ---
        hours, rem = divmod(training_duration, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.info("\n--- Training Summary ---")
        logger.info(f"Total training runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
        
        if total_flops > 0 and flops_analyzer is not None:
            logger.info(f"Total FLOPs per forward pass: {total_flops / 1e9:.2f} GFLOPs")
            # --- More explicit FLOPs breakdown ---
            logger.info("--- FLOPs Breakdown by Module ---")
            logger.info(flops_analyzer.by_module())
            logger.info("---------------------------------")
        logger.info("------------------------\n")
        # --- End Reporting ---

        # --- Plotting Epoch Times ---
        if epoch_timer_callback.epoch_times:
            logger.info("Generating plot for epoch training times...")
            try:
                plt.figure(figsize=(10, 6))
                num_epochs_completed = range(1, len(epoch_timer_callback.epoch_times) + 1)
                plt.plot(num_epochs_completed, epoch_timer_callback.epoch_times, marker='o', linestyle='-')
                plt.title('Training Time per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Time (seconds)')
                plt.grid(True)
                # Ensure x-axis ticks are integers
                plt.xticks(list(num_epochs_completed))
                plt.tight_layout()
                plot_filename = 'epoch_times.png'
                plt.savefig(plot_filename)
                plt.close() # Close the figure to free up memory
                logger.info(f"Epoch times plot saved to {plot_filename}")
            except Exception as e:
                logger.error(f"Failed to generate epoch times plot: {e}")
        # --- End Plotting ---

        logger.info("Cleaning up resources...")
        wandb.finish()

        del model
        del data_module
        del trainer
        del visualizer_callback

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")

        logger.info("Cleanup complete. Exiting.")

if __name__ == '__main__':
    main_HiMAE()