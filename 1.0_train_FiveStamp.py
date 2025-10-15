#!/usr/bin/env python3
# train_gan_with_anchor.py
# GAN (LSTM) for facial+neck motion generation with anchor guidance (Mean Anchor Loss).
# Generator contains emotion embedding (so generator_final.pth is sufficient for deployment).
# Saves final plots in ./plot/ and models in ./model/.

import os
import glob
import json
import random
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import math
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# User-editable hyperparameters
# ----------------------------
DATA_DIR = "train_data"                # folder with merge_*.csv files
ANCHOR_CSV = "expression_anchor.csv"   # anchor csv (first col label, next 29 columns face params)
# 获取当前时间戳（如 20251015_1735）
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
# 创建主输出文件夹 output/
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
# 创建本次运行的子文件夹，如 output/20251015_1735/
RUN_DIR = os.path.join(OUTPUT_ROOT, timestamp)
os.makedirs(RUN_DIR, exist_ok=True)
MODEL_DIR = RUN_DIR
PLOT_DIR = RUN_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 32      # 29 face + 3 neck
FACE_DIM = 29
SEQ_LEN = 20
PRED_LEN = 5
NUM_EMOTIONS = 7
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMO_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

BATCH_SIZE = 32
NUM_EPOCHS = 1200    # default; you can reduce for testing

# separate learning rates for G and D
LR_G = 1e-4
LR_D = 1e-5
BETAS = (0.5, 0.999)

# loss weights (target values)
LAMBDA_ADV = 0.01
LAMBDA_REC = 1.0
LAMBDA_ANCHOR = 0.8

PRINT_EVERY = 50

# Pretrain / schedule settings
PRETRAIN_FRAC = 1/3.0   # first 1/3 epochs: pretrain G only (rec + anchor)
WARMUP_FRAC = 1/3.0     # next 1/3 epochs: adversarial with lambda_adv increasing
# last part uses full LAMBDA_ADV

# instance noise schedule (applied to D inputs during adversarial training)
NOISE_START = 0.08
NOISE_END = 0.0

# misc
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# ----------------------------
# Data loading utilities
# ----------------------------
def load_all_label_data(data_dir: str, emotions: List[str]) -> Dict[str, np.ndarray]:
    data = {}
    missing_files = []
    all_files = os.listdir(data_dir)
    for emo in emotions:
        # match pattern: merged_..._{emo}.csv (robust to prefixes)
        matched_files = [os.path.join(data_dir, f) for f in all_files if f.startswith("merge") and f.endswith(f"_{emo}.csv")]
        if not matched_files:
            print(f"Warning: no file found for emotion '{emo}'")
            missing_files.append(emo)
            continue
        frames_list = []
        for fp in matched_files:
            df = pd.read_csv(fp, header=None)
            arr = df.values.astype(np.float32)
            if arr.shape[1] < INPUT_DIM:
                raise ValueError(f"File {fp} has {arr.shape[1]} columns < required INPUT_DIM={INPUT_DIM}")
            frames_list.append(arr[:, :INPUT_DIM])
        if frames_list:
            data[emo] = np.vstack(frames_list)
            print(f"Loaded {data[emo].shape[0]} frames for emotion '{emo}' from {len(matched_files)} file(s).")
    if missing_files:
        print(f"Missing data for emotions: {missing_files}")
    return data


def get_exponential_noise_sigma(epoch_idx, total_adv_epochs, noise_start, noise_end, decay_rate=5.0):
    """
    Exponential decay of instance noise.
    sigma = noise_start * exp(-decay_rate * progress), clipped to >= noise_end
    progress ∈ [0,1]
    """
    progress = epoch_idx / max(1, total_adv_epochs - 1)
    sigma = noise_start * math.exp(-decay_rate * progress)
    return max(sigma, noise_end)


class SlidingWindowDataset(Dataset):
    """
    Preloads per-emotion arrays and enumerates all sliding windows across all emotions.
    Each item is (input_seq (SEQ_LEN x INPUT_DIM), target_seq (PRED_LEN x INPUT_DIM), label_idx)
    """
    def __init__(self, data_by_label: Dict[str, np.ndarray], emotions: List[str],
                 seq_len: int = SEQ_LEN, pred_len: int = PRED_LEN):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.emotions = emotions
        self.data = {}
        self.index_map = []  # list of tuples (emotion, start_idx)
        for emo in emotions:
            if emo in data_by_label:
                arr = data_by_label[emo]
                n = arr.shape[0]
                # we require at least seq_len + pred_len frames to produce at least one window
                if n >= seq_len + pred_len:
                    self.data[emo] = arr
                    max_start = n - (seq_len + pred_len)
                    for s in range(max_start + 1):
                        self.index_map.append((emo, s))
                else:
                    print(f"Warning: emotion '{emo}' has {n} frames < required {seq_len+pred_len}. Skipping.")
            else:
                # no data for this emotion
                pass
        if len(self.index_map) == 0:
            raise RuntimeError("No sliding windows found. Check your data files and lengths.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        emo, start = self.index_map[idx]
        arr = self.data[emo]
        seq = arr[start:start + self.seq_len, :INPUT_DIM].astype(np.float32)
        target = arr[start + self.seq_len:start + self.seq_len + self.pred_len, :INPUT_DIM].astype(np.float32)
        label_idx = EMO_TO_IDX[emo]
        return torch.from_numpy(seq), torch.from_numpy(target), torch.tensor(label_idx, dtype=torch.long)


# ----------------------------
# Anchor loader
# ----------------------------
def load_anchors(anchor_csv: str) -> Dict[str, torch.Tensor]:
    """
    Reads anchor CSV. Expect header row (column names), first column that contains label name (e.g., 'label' or 'expression'),
    following columns contain face dims (29 columns). Returns a dict emotion->tensor(29,).
    """
    if not os.path.exists(anchor_csv):
        raise FileNotFoundError(f"Anchor CSV '{anchor_csv}' not found.")
    df = pd.read_csv(anchor_csv, header=0)
    label_col = df.columns[0]
    anchors = {}
    for _, row in df.iterrows():
        emo = str(row[label_col]).strip()
        vals = row.values[1:1 + FACE_DIM].astype(np.float32)
        if vals.shape[0] != FACE_DIM:
            raise ValueError(f"Anchor row for '{emo}' does not have {FACE_DIM} face values.")
        anchors[emo] = torch.from_numpy(vals).float()
    for emo in EMOTIONS:
        if emo not in anchors:
            print(f"Warning: no anchor found for emotion '{emo}'. It will be skipped in anchor loss.")
    return anchors


# ----------------------------
# Models: Generator (with embedding) & Discriminator (LSTM)
# ----------------------------
class GeneratorLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, embed_dim=4, hidden_dim=256, num_layers=2, pred_len=PRED_LEN, num_emotions=NUM_EMOTIONS):
        super().__init__()
        self.embed = nn.Embedding(num_emotions, embed_dim)
        self.lstm = nn.LSTM(input_dim + embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, pred_len * input_dim)
        )
        self.pred_len = pred_len
        self.input_dim = input_dim

    def forward(self, x, labels):
        # x: (B, seq_len, input_dim); labels: (B,)
        emb = self.embed(labels)                    # (B, embed_dim)
        emb_exp = emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, seq_len, embed_dim)
        inp = torch.cat([x, emb_exp], dim=2)        # (B, seq_len, input_dim+embed)
        out, (h_n, c_n) = self.lstm(inp)  # out: (B, seq_len, hidden)
        seq_feature = out.mean(dim=1)      # 对时间维度取平均 (B, hidden)
        fc_out = self.fc(seq_feature)      # (B, pred_len * input_dim)
        fc_out = fc_out.view(-1, self.pred_len, self.input_dim)
        return fc_out


class DiscriminatorLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, seq):
        # seq: (B, total_seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(seq)
        seq_feature = out.mean(dim=1)
        logits = self.fc(seq_feature).view(-1)
        return logits  # raw logits


# ----------------------------
# Loss helpers (Hinge)
# ----------------------------
def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(torch.relu(1.0 - real_logits))
    loss_fake = torch.mean(torch.relu(1.0 + fake_logits))
    return loss_real + loss_fake


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -torch.mean(fake_logits)


def mean_anchor_loss(pred_seq: torch.Tensor, labels: torch.Tensor, anchors: Dict[str, torch.Tensor]) -> torch.Tensor:
    device = pred_seq.device
    losses = []
    for i in range(pred_seq.size(0)):
        emo = EMOTIONS[labels[i].item()]
        if emo not in anchors:
            continue
        anchor_vec = anchors[emo].to(device)  # (FACE_DIM,)
        face_mean = torch.mean(pred_seq[i, :, :FACE_DIM], dim=0)  # (FACE_DIM,)
        losses.append(torch.mean((face_mean - anchor_vec) ** 2))
    if len(losses) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


# ----------------------------
# Instance noise helper
# ----------------------------
def add_instance_noise(x: torch.Tensor, sigma: float):
    if sigma <= 0.0:
        return x
    return x + sigma * torch.randn_like(x)


# ----------------------------
# Training routine
# ----------------------------
def train():
    # schedule
    pretrain_epochs = int(math.floor(NUM_EPOCHS * PRETRAIN_FRAC))
    warmup_epochs = int(math.floor(NUM_EPOCHS * WARMUP_FRAC))
    final_epochs = NUM_EPOCHS - pretrain_epochs - warmup_epochs
    print(f"Schedule: pretrain {pretrain_epochs} epochs, warmup {warmup_epochs} epochs, final {final_epochs} epochs")

    # load data
    print("Loading data from:", DATA_DIR)
    data_by_label = load_all_label_data(DATA_DIR, EMOTIONS)
    dataset = SlidingWindowDataset(data_by_label, EMOTIONS, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    print(f"Total sliding windows: {len(dataset)}; batches per epoch: {len(dataloader)}")

    # anchors
    anchors = load_anchors(ANCHOR_CSV)
    if anchors:
        print("Anchors loaded for:", list(anchors.keys()))

    # models
    G = GeneratorLSTM().to(DEVICE)
    D = DiscriminatorLSTM().to(DEVICE)

    # optimizers
    g_opt = optim.Adam(G.parameters(), lr=LR_G, betas=BETAS)
    d_opt = optim.Adam(D.parameters(), lr=LR_D, betas=BETAS)

    l1_loss = nn.L1Loss()

    # metrics history
    history = {
        "G_loss": [], "D_loss": [], "adv_loss": [], "rec_loss": [], "anchor_loss": [],
        "D_real_mean": [], "D_fake_mean": []
    }

    global_step = 0
    epoch_counter = 0

    # -----------------
    # 1) Pretrain G (behavior cloning): only rec + anchor
    # -----------------
    if pretrain_epochs > 0:
        print("=== Pretraining generator (behavior cloning) ===")
        for epoch in range(1, pretrain_epochs + 1):
            G.train()
            epoch_g_loss = 0.0
            epoch_rec = 0.0
            epoch_anchor = 0.0
            for batch_idx, (input_seq, target_seq, label_idx) in enumerate(dataloader):
                input_seq = input_seq.to(DEVICE)
                target_seq = target_seq.to(DEVICE)
                label_idx = label_idx.to(DEVICE)

                fake_pred = G(input_seq, label_idx)
                rec_loss = l1_loss(fake_pred, target_seq)
                anchor_loss = mean_anchor_loss(fake_pred, label_idx, anchors)
                g_loss = LAMBDA_REC * rec_loss + LAMBDA_ANCHOR * anchor_loss

                g_opt.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                g_opt.step()

                epoch_g_loss += g_loss.item()
                epoch_rec += rec_loss.item()
                epoch_anchor += anchor_loss.item() if isinstance(anchor_loss, torch.Tensor) else float(anchor_loss)
                global_step += 1

                if (global_step % PRINT_EVERY) == 0:
                    print(f"[Pretrain Epoch {epoch}/{pretrain_epochs}] Step[{batch_idx}/{len(dataloader)}] g_loss={g_loss.item():.4f} rec={rec_loss.item():.4f} anchor={anchor_loss:.6f}")

            n_batches = len(dataloader)
            history["G_loss"].append(epoch_g_loss / n_batches)
            history["rec_loss"].append(epoch_rec / n_batches)
            history["anchor_loss"].append(epoch_anchor / n_batches)
            history["D_real_mean"].append(0.0)
            history["D_fake_mean"].append(0.0)
            history["D_loss"].append(0.0)
            history["adv_loss"].append(0.0)
            history["D_real_mean"].append(0.0)
            history["D_fake_mean"].append(0.0)

            # 每100个epoch保存一次模型
            if epoch % 100 == 0:
                gen_path = os.path.join(MODEL_DIR, f"generator_epoch_{epoch}.pth")
                disc_path = os.path.join(MODEL_DIR, f"discriminator_epoch_{epoch}.pth")
                torch.save(G.state_dict(), gen_path)
                torch.save(D.state_dict(), disc_path)
                print(f"Saved models at epoch {epoch}: {gen_path}, {disc_path}")

            epoch_counter += 1
            print(f"Pretrain Epoch {epoch} summary: G={history['G_loss'][-1]:.4f} rec={history['rec_loss'][-1]:.4f} anchor={history['anchor_loss'][-1]:.4f}")

    # -----------------
    # 2) Warmup adversarial: lambda_adv linearly increases from adv_start to LAMBDA_ADV
    # -----------------
    adv_start = 0.1  # starting adv weight during warmup (small)
    if warmup_epochs > 0:
        print("=== Warmup adversarial training (lambda_adv increasing) ===")
        for epoch in range(1, warmup_epochs + 1):
            curr_epoch_global = epoch_counter + epoch
            # linearly increase lambda_adv from adv_start -> LAMBDA_ADV over warmup_epochs
            frac = (epoch - 1) / max(1, warmup_epochs - 1)
            curr_lambda_adv = adv_start + frac * (LAMBDA_ADV - adv_start)

            G.train(); D.train()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_adv = 0.0
            epoch_rec = 0.0
            epoch_anchor = 0.0
            epoch_d_real_mean = 0.0
            epoch_d_fake_mean = 0.0

            # noise schedule: anneal from NOISE_START -> NOISE_END across warmup + final epochs
            total_adv_epochs = warmup_epochs + final_epochs
            adv_epoch_index = epoch - 1  # 0-based in warmup
            # compute sigma for this epoch relative to whole adversarial duration
            adv_progress = adv_epoch_index / max(1, total_adv_epochs - 1)
            sigma = get_exponential_noise_sigma(adv_epoch_index, total_adv_epochs, NOISE_START, NOISE_END)

            for batch_idx, (input_seq, target_seq, label_idx) in enumerate(dataloader):
                input_seq = input_seq.to(DEVICE)
                target_seq = target_seq.to(DEVICE)
                label_idx = label_idx.to(DEVICE)

                # -------------------- D step --------------------
                with torch.no_grad():
                    fake_pred = G(input_seq, label_idx)
                real_seq = torch.cat([input_seq, target_seq], dim=1)
                fake_seq = torch.cat([input_seq, fake_pred], dim=1)

                # add instance noise
                real_seq_noisy = add_instance_noise(real_seq, sigma)
                fake_seq_noisy = add_instance_noise(fake_seq, sigma)

                real_logits = D(real_seq_noisy)
                fake_logits = D(fake_seq_noisy)

                d_loss = discriminator_hinge_loss(real_logits, fake_logits)
                d_opt.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                d_opt.step()

                # -------------------- G step --------------------
                fake_pred = G(input_seq, label_idx)
                fake_seq_for_d = torch.cat([input_seq, fake_pred], dim=1)
                fake_seq_for_d_noisy = add_instance_noise(fake_seq_for_d, sigma)
                fake_logits_for_g = D(fake_seq_for_d_noisy)

                adv_loss = generator_hinge_loss(fake_logits_for_g)
                rec_loss = l1_loss(fake_pred, target_seq)
                anchor_loss = mean_anchor_loss(fake_pred, label_idx, anchors)

                g_loss = curr_lambda_adv * adv_loss + LAMBDA_REC * rec_loss + LAMBDA_ANCHOR * anchor_loss

                g_opt.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                g_opt.step()

                # stats
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_adv += adv_loss.item()
                epoch_rec += rec_loss.item()
                epoch_anchor += anchor_loss.item() if isinstance(anchor_loss, torch.Tensor) else float(anchor_loss)
                epoch_d_real_mean += real_logits.mean().item()
                epoch_d_fake_mean += fake_logits.mean().item()
                global_step += 1

                if (global_step % PRINT_EVERY) == 0:
                    print(f"[Warmup Epoch {epoch}/{warmup_epochs}] Step[{batch_idx}/{len(dataloader)}] lambda_adv={curr_lambda_adv:.4f} D_loss={d_loss.item():.4f} G_loss={g_loss.item():.4f} adv={adv_loss.item():.4f} rec={rec_loss.item():.4f} anchor={anchor_loss:.6f} sigma={sigma:.4f}")

            # record epoch stats
            n_batches = len(dataloader)
            history["G_loss"].append(epoch_g_loss / n_batches)
            history["D_loss"].append(epoch_d_loss / n_batches)
            history["adv_loss"].append(epoch_adv / n_batches)
            history["rec_loss"].append(epoch_rec / n_batches)
            history["anchor_loss"].append(epoch_anchor / n_batches)
            history["D_real_mean"].append(epoch_d_real_mean / n_batches)
            history["D_fake_mean"].append(epoch_d_fake_mean / n_batches)
            
            # 每100个epoch保存一次模型
            if curr_epoch_global % 100 == 0:
                gen_path = os.path.join(MODEL_DIR, f"generator_epoch_{curr_epoch_global}.pth")
                disc_path = os.path.join(MODEL_DIR, f"discriminator_epoch_{curr_epoch_global}.pth")
                torch.save(G.state_dict(), gen_path)
                torch.save(D.state_dict(), disc_path)
                print(f"Saved models at epoch {curr_epoch_global}: {gen_path}, {disc_path}")
            
            epoch_counter += 1

            print(f"Warmup Epoch {curr_epoch_global} summary: lambda_adv={curr_lambda_adv:.4f} G={history['G_loss'][-1]:.4f} D={history['D_loss'][-1]:.4f} adv={history['adv_loss'][-1]:.4f} rec={history['rec_loss'][-1]:.4f} anchor={history['anchor_loss'][-1]:.4f} D_real={history['D_real_mean'][-1]:.4f} D_fake={history['D_fake_mean'][-1]:.4f}")

    # -----------------
    # 3) Final adversarial: use full LAMBDA_ADV, continue annealing noise
    # -----------------
    if final_epochs > 0:
        print("=== Final adversarial training (full lambda_adv) ===")
        for epoch in range(1, final_epochs + 1):
            curr_epoch_global = epoch_counter + epoch
            curr_lambda_adv = LAMBDA_ADV
            # epoch progress across final portion and previous warmup to compute sigma annealing
            total_adv_epochs = warmup_epochs + final_epochs
            adv_epoch_index = warmup_epochs + (epoch - 1)
            adv_progress = adv_epoch_index / max(1, total_adv_epochs - 1)
            sigma = get_exponential_noise_sigma(adv_epoch_index, total_adv_epochs, NOISE_START, NOISE_END)

            G.train(); D.train()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_adv = 0.0
            epoch_rec = 0.0
            epoch_anchor = 0.0
            epoch_d_real_mean = 0.0
            epoch_d_fake_mean = 0.0

            for batch_idx, (input_seq, target_seq, label_idx) in enumerate(dataloader):
                input_seq = input_seq.to(DEVICE)
                target_seq = target_seq.to(DEVICE)
                label_idx = label_idx.to(DEVICE)

                # D step
                with torch.no_grad():
                    fake_pred = G(input_seq, label_idx)
                real_seq = torch.cat([input_seq, target_seq], dim=1)
                fake_seq = torch.cat([input_seq, fake_pred], dim=1)
                real_seq_noisy = add_instance_noise(real_seq, sigma)
                fake_seq_noisy = add_instance_noise(fake_seq, sigma)
                real_logits = D(real_seq_noisy)
                fake_logits = D(fake_seq_noisy)
                d_loss = discriminator_hinge_loss(real_logits, fake_logits)
                d_opt.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                d_opt.step()

                # G step
                fake_pred = G(input_seq, label_idx)
                fake_seq_for_d = torch.cat([input_seq, fake_pred], dim=1)
                fake_seq_for_d_noisy = add_instance_noise(fake_seq_for_d, sigma)
                fake_logits_for_g = D(fake_seq_for_d_noisy)
                adv_loss = generator_hinge_loss(fake_logits_for_g)
                rec_loss = l1_loss(fake_pred, target_seq)
                anchor_loss = mean_anchor_loss(fake_pred, label_idx, anchors)
                g_loss = curr_lambda_adv * adv_loss + LAMBDA_REC * rec_loss + LAMBDA_ANCHOR * anchor_loss
                g_opt.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                g_opt.step()

                # stats
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_adv += adv_loss.item()
                epoch_rec += rec_loss.item()
                epoch_anchor += anchor_loss.item() if isinstance(anchor_loss, torch.Tensor) else float(anchor_loss)
                epoch_d_real_mean += real_logits.mean().item()
                epoch_d_fake_mean += fake_logits.mean().item()
                global_step += 1

                if (global_step % PRINT_EVERY) == 0:
                    print(f"[Final Epoch {epoch}/{final_epochs}] Step[{batch_idx}/{len(dataloader)}] D_loss={d_loss.item():.4f} G_loss={g_loss.item():.4f} adv={adv_loss.item():.4f} rec={rec_loss.item():.4f} anchor={anchor_loss:.6f} sigma={sigma:.4f}")

            # record epoch stats
            n_batches = len(dataloader)
            history["G_loss"].append(epoch_g_loss / n_batches)
            history["D_loss"].append(epoch_d_loss / n_batches)
            history["adv_loss"].append(epoch_adv / n_batches)
            history["rec_loss"].append(epoch_rec / n_batches)
            history["anchor_loss"].append(epoch_anchor / n_batches)
            history["D_real_mean"].append(epoch_d_real_mean / n_batches)
            history["D_fake_mean"].append(epoch_d_fake_mean / n_batches)
            
            # 每100个epoch保存一次模型
            if curr_epoch_global % 100 == 0:
                gen_path = os.path.join(MODEL_DIR, f"generator_epoch_{curr_epoch_global}.pth")
                disc_path = os.path.join(MODEL_DIR, f"discriminator_epoch_{curr_epoch_global}.pth")
                torch.save(G.state_dict(), gen_path)
                torch.save(D.state_dict(), disc_path)
                print(f"Saved models at epoch {curr_epoch_global}: {gen_path}, {disc_path}")
            
            epoch_counter += 1

            print(f"Final Epoch {curr_epoch_global} summary: G={history['G_loss'][-1]:.4f} D={history['D_loss'][-1]:.4f} adv={history['adv_loss'][-1]:.4f} rec={history['rec_loss'][-1]:.4f} anchor={history['anchor_loss'][-1]:.4f} D_real={history['D_real_mean'][-1]:.4f} D_fake={history['D_fake_mean'][-1]:.4f}")

    # ----------------------------
    # After training: save final models and plots
    # ----------------------------
    gen_path = os.path.join(MODEL_DIR, "generator.pth")
    disc_path = os.path.join(MODEL_DIR, "discriminator.pth")
    torch.save(G.state_dict(), gen_path)
    torch.save(D.state_dict(), disc_path)
    print(f"Saved models: {gen_path}, {disc_path}")

    # Loss curve
    plt.figure(figsize=(8,5))
    plt.plot(history.get("G_loss", []), label="G_loss")
    plt.plot(history.get("D_loss", []), label="D_loss")
    if len(history.get("adv_loss", [])) > 0:
        plt.plot(history["adv_loss"], label="adv_loss")
    if len(history.get("rec_loss", [])) > 0:
        plt.plot(history["rec_loss"], label="rec_loss")
    if len(history.get("anchor_loss", [])) > 0:
        plt.plot(history["anchor_loss"], label="anchor_loss")
    plt.xlabel("Epoch (phase epochs concatenated)")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "loss_curve_final.png"), dpi=200)
    plt.close()
    print(f"Saved loss curve to {os.path.join(PLOT_DIR, 'loss_curve_final.png')}")

    # Discriminator output distribution: sample random windows to compute real/fake logits hist
    with torch.no_grad():
        vis_idx = np.random.choice(len(dataset), size=min(256, len(dataset)), replace=False)
        real_logits_list = []
        fake_logits_list = []
        for i in range(0, len(vis_idx), BATCH_SIZE):
            batch_idx = vis_idx[i:i+BATCH_SIZE]
            items = [dataset[j] for j in batch_idx]
            seqs = torch.stack([it[0] for it in items]).to(DEVICE)
            targs = torch.stack([it[1] for it in items]).to(DEVICE)
            labels = torch.stack([it[2] for it in items]).to(DEVICE)
            real_seq = torch.cat([seqs, targs], dim=1)
            fake_pred = G(seqs, labels)
            fake_seq = torch.cat([seqs, fake_pred], dim=1)
            # use no noise for final viz
            real_logits = D(real_seq).cpu().numpy()
            fake_logits = D(fake_seq).cpu().numpy()
            real_logits_list.append(real_logits)
            fake_logits_list.append(fake_logits)
        real_all = np.concatenate(real_logits_list)
        fake_all = np.concatenate(fake_logits_list)

    plt.figure(figsize=(8,5))
    plt.hist(real_all, bins=60, alpha=0.6, label='D(real)')
    plt.hist(fake_all, bins=60, alpha=0.6, label='D(fake)')
    plt.legend()
    plt.title("Discriminator Output Distribution (real vs fake)")
    plt.xlabel("Logit value")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "disc_output_distribution.png"), dpi=200)
    plt.close()
    print(f"Saved discriminator distribution to {os.path.join(PLOT_DIR, 'disc_output_distribution.png')}")

    # optionally save history metadata
    meta = {
        "params": {
            "SEQ_LEN": SEQ_LEN, "PRED_LEN": PRED_LEN, "INPUT_DIM": INPUT_DIM,
            "BATCH_SIZE": BATCH_SIZE, "NUM_EPOCHS": NUM_EPOCHS, "LR_G": LR_G, "LR_D": LR_D,
            "LAMBDA_ADV": LAMBDA_ADV, "LAMBDA_REC": LAMBDA_REC, "LAMBDA_ANCHOR": LAMBDA_ANCHOR
        },
        "emotions": EMOTIONS
    }
    with open(os.path.join(MODEL_DIR, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Training finished. Models, plots and meta saved.")


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    # quick checks
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"train_data directory '{DATA_DIR}' not found. Put your merge CSVs there.")
    if not os.path.exists(ANCHOR_CSV):
        raise RuntimeError(f"Anchor CSV '{ANCHOR_CSV}' not found. Place expression_anchor.csv in script directory.")
    start_time = time.time()
    train()
    print(f"Total run time: {time.time() - start_time:.1f}s")
