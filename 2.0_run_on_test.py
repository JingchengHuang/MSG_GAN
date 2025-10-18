#!/usr/bin/env python3
# 2.0_run_on_test_fixed.py
# Inference script for trained generator model with sliding window and diagonal averaging
# Implements the averaging pattern described by the user:
# step t uses pred_t[0], pred_{t-1}[1], pred_{t-2}[2], ... up to PRED_LEN terms.

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import glob
import re
from typing import Dict, List

# ----------------------------
# Hyperparameters (should match training)
# ----------------------------
INPUT_DIM = 32      # 29 face + 3 neck
FACE_DIM = 29       # face parameters
NECK_DIM = 3        # neck parameters
SEQ_LEN = 20
PRED_LEN = 5
NUM_EMOTIONS = 7
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMO_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neck parameter ranges for denormalization
NECK_RANGES = {
    29: (-0.8, 0.6),   #倒数第三个数
    30: (-0.3, 0.3),   #倒数第二个数
    31: (-0.55, 0.55)  #倒数第一个数
}

# ----------------------------
# Generator Model (should match training)
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

# ----------------------------
# Data loading utilities
# ----------------------------
def load_label_data(data_dir: str, emotion: str) -> np.ndarray:
    """
    Load data for a specific emotion from train_data directory.
    Returns ndarray (N_frames x INPUT_DIM)
    """
    all_files = os.listdir(data_dir)
    # match pattern: merged_..._{emotion}.csv (robust to prefixes)
    matched_files = [os.path.join(data_dir, f) for f in all_files if f.startswith("merge") and f.endswith(f"_{emotion}.csv")]
    if not matched_files:
        raise FileNotFoundError(f"No file found for emotion '{emotion}' in {data_dir}")
    
    frames_list = []
    for fp in matched_files:
        df = pd.read_csv(fp, header=None)
        arr = df.values.astype(np.float32)
        if arr.shape[1] < INPUT_DIM:
            raise ValueError(f"File {fp} has {arr.shape[1]} columns < required INPUT_DIM={INPUT_DIM}")
        frames_list.append(arr[:, :INPUT_DIM])
    
    data = np.vstack(frames_list)
    print(f"Loaded {data.shape[0]} frames for emotion '{emotion}' from {len(matched_files)} file(s).")
    return data

def denormalize_neck_params(params: np.ndarray) -> np.ndarray:
    """
    Denormalize neck parameters from [0,1] to their respective ranges.
    params: ndarray of shape (..., INPUT_DIM) in normalized space
    Returns denormalized params with same shape
    """
    params = params.copy()
    # Denormalize neck parameters (indices 29, 30, 31)
    for i in range(29, 32):
        min_val, max_val = NECK_RANGES[i]
        params[..., i] = params[..., i] * (max_val - min_val) + min_val
    return params

def clamp_params(params: np.ndarray) -> np.ndarray:
    """
    Clamp parameters to valid ranges in normalized space.
    Face params (0-28): clamped to [0, 1]
    Neck params (29-31): assume in normalized [0,1] space for now (no clamp)
    params: ndarray of shape (..., INPUT_DIM)
    Returns clamped params with same shape
    """
    params = params.copy()
    # Clamp face parameters (0-28) to [0, 1]
    params[..., :29] = np.clip(params[..., :29], 0.0, 1.0)
    # Ensure neck normalized values are also clipped to [0,1] to avoid bad denormalization input
    params[..., 29:32] = np.clip(params[..., 29:32], 0.0, 1.0)
    return params

def check_param_ranges(params: np.ndarray, step: int = None) -> bool:
    """
    Check if all parameters are within valid ranges.
    params expected to be DENORMALIZED for neck (i.e., neck in actual physical ranges),
    and face params in [0,1].
    Returns True if all parameters are valid, False otherwise.
    """
    # Check face parameters (0-28): should be in [0, 1]
    face_params = params[..., :29]
    if np.any(face_params < 0.0) or np.any(face_params > 1.0):
        if step is not None:
            print(f"Error at step {step}: Face parameters out of range [0, 1]")
        else:
            print("Error: Face parameters out of range [0, 1]")
        print(f"Face params min: {face_params.min()}, max: {face_params.max()}")
        return False
    
    # Check neck parameters (29-31): should be in their respective ranges (DENORMALIZED)
    for i in range(29, 32):
        min_val, max_val = NECK_RANGES[i]
        neck_param = params[..., i]
        if np.any(neck_param < min_val) or np.any(neck_param > max_val):
            if step is not None:
                print(f"Error at step {step}: Neck parameter {i} out of range [{min_val}, {max_val}]")
            else:
                print(f"Error: Neck parameter {i} out of range [{min_val}, {max_val}]")
            print(f"Neck param {i} min: {neck_param.min()}, max: {neck_param.max()}")
            return False
    
    return True

def get_random_slice(data: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Get a random slice of seq_len frames from data.
    Returns ndarray (seq_len x INPUT_DIM) in normalized space
    """
    if data.shape[0] < seq_len:
        raise ValueError(f"Data has {data.shape[0]} frames < required seq_len={seq_len}")
    
    max_start = data.shape[0] - seq_len
    start_idx = random.randint(0, max_start)
    
    # Get slice and apply clamping
    slice_data = data[start_idx:start_idx + seq_len]
    slice_data = clamp_params(slice_data)
    
    return slice_data

# ----------------------------
# Main inference function
# ----------------------------
def run_inference(emotion: str, max_steps: int = 100, model_path: str = "model/generator.pth"):
    """
    Run inference with sliding window and diagonal averaging as described.
    
    Args:
        emotion: emotion label to use for inference
        max_steps: maximum number of inference steps
        model_path: path to trained generator model
    """
    # Load data for specified emotion
    print(f"Loading data for emotion: {emotion}")
    data = load_label_data("train_data", emotion)
    
    # Initialize model
    print("Initializing model...")
    model = GeneratorLSTM().to(DEVICE)
    
    # Load trained model weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model from: {model_path}")
    
    # Get initial sliding window (20 frames) -- in normalized space
    print("Getting initial sliding window...")
    initial_window = get_random_slice(data, SEQ_LEN)  # (SEQ_LEN x INPUT_DIM) normalized
    sliding_window = initial_window.copy()  # (SEQ_LEN x INPUT_DIM)
    
    # Emotion label tensor
    emotion_idx = EMO_TO_IDX[emotion]
    label_tensor = torch.tensor([emotion_idx], dtype=torch.long).to(DEVICE)  # (1,)
    
    print(f"Starting inference for {max_steps} steps...")
    print(f"Initial window shape: {sliding_window.shape}")
    
    # Store previous predictions (in normalized space) for diagonal averaging
    prev_predictions: List[np.ndarray] = []  # each element shape: (PRED_LEN, INPUT_DIM), normalized
    # To bound memory we can cap stored history (we only need up to PRED_LEN past ones for averaging),
    # but keep a few extra in case. We'll keep last 100 by default.
    MAX_HISTORY = 100
    
    with torch.no_grad():
        for step in range(max_steps):
            # Convert sliding window to tensor (normalized)
            window_tensor = torch.from_numpy(sliding_window).unsqueeze(0).float().to(DEVICE)  # (1, SEQ_LEN, INPUT_DIM)
            
            # Run inference -> returns normalized prediction (assuming model trained to output normalized values)
            prediction = model(window_tensor, label_tensor)  # (1, PRED_LEN, INPUT_DIM)
            prediction = prediction.squeeze(0).cpu().numpy()  # (PRED_LEN, INPUT_DIM) normalized (expected)
            
            # Clamp prediction in normalized space (face in [0,1], neck normalized to [0,1])
            prediction = clamp_params(prediction)
            
            # Build diagonal-average frame in normalized space:
            # frames_to_avg = [ prediction[0], prev_predictions[-1][1], prev_predictions[-2][2], ... ]
            frames = []
            # current prediction's first frame
            frames.append(prediction[0])
            # gather previous predictions if available
            for k in range(1, PRED_LEN):
                # need at least k previous predictions to take their k-th frame (index k)
                if len(prev_predictions) >= k:
                    # prev_predictions[-k] is the prediction from k steps ago (1-index)
                    # we want its frame with index k (0-indexed)
                    cand_pred = prev_predictions[-k]
                    if cand_pred.shape[0] > k:
                        frames.append(cand_pred[k])
                    else:
                        # if candidate prediction too short (shouldn't happen) skip
                        pass
                else:
                    break
            
            # Now compute averaged normalized frame
            averaged_frame_norm = np.mean(np.stack(frames, axis=0), axis=0)  # (INPUT_DIM,)
            # Ensure normalized values in valid normalized ranges
            averaged_frame_norm = clamp_params(averaged_frame_norm.reshape(1, -1)).flatten()
            
            # Denormalize neck parameters for use and final checks
            averaged_frame_denorm = denormalize_neck_params(averaged_frame_norm.reshape(1, -1)).flatten()
            
            # Check ranges for averaged frame (face in [0,1], neck denormalized)
            if not check_param_ranges(averaged_frame_denorm.reshape(1, -1), step+1):
                raise ValueError(f"Averaged frame at step {step+1} contains parameters out of valid ranges")
            
            # Construct new sliding window: drop oldest, append averaged_frame_denorm (face still expected in [0,1], neck denorm)
            # BUT our sliding_window is maintained in normalized space. So we should append the normalized averaged frame.
            new_window = np.vstack([sliding_window[1:], averaged_frame_norm])  # (SEQ_LEN, INPUT_DIM) normalized
            
            # As a safeguard, clamp new window
            new_window = clamp_params(new_window)
            
            # For any internal checks that expect denormalized neck, prepare a denormalized copy for checking
            new_window_denorm = denormalize_neck_params(new_window.copy())
            if not check_param_ranges(new_window_denorm, step+1):
                raise ValueError(f"New window at step {step+1} contains parameters out of valid ranges")
            
            # Update sliding window (keep normalized representation)
            sliding_window = new_window
            
            # Print the last frame of updated sliding window (denormalized for neck readability)
            # We'll print face (normalized) and neck (denormalized)
            to_print = sliding_window[-1].copy()
            to_print_denorm = denormalize_neck_params(to_print.reshape(1, -1)).flatten()
            print(f"Step {step+1}: {to_print_denorm}")
            
            # Store current prediction (normalized) for future averaging
            prev_predictions.append(prediction.copy())  # prediction already normalized & clamped
            # Trim history to bounded size
            if len(prev_predictions) > MAX_HISTORY:
                prev_predictions = prev_predictions[-MAX_HISTORY:]
            
            # Optional: Print some debug info
            if (step + 1) % 10 == 0:
                print(f"Completed {step + 1}/{max_steps} steps")
    
    print("Inference completed.")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    # User-configurable parameters
    EMOTION = "happy"      # Change this to desired emotion: 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
    MAX_STEPS = 500        # Change this to desired number of inference steps
    MODEL_PATH = "model/generator.pth"  # Path to trained generator model
    
    # Check if data directory exists
    if not os.path.isdir("train_data"):
        raise RuntimeError("train_data directory not found. Please ensure data files are in the correct location.")
    
    run_inference(EMOTION, MAX_STEPS, MODEL_PATH)
