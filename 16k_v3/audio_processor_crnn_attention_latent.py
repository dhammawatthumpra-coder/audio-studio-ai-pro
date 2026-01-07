"""
CRNN + GRU + Attention 16kHz Native Speech Enhancement
Version: 5.0 (CRNN+GRU+Attention - Ultimate Quality)
Features: Native 16k, 8k Control, CRNN+GRU+Attention (99K params)
Date: 2026-01-05

Uses CRNN+GRU+Attention for ultimate audio quality.
- CNN: Spectral features
- GRU: Temporal dynamics  
- Attention: Focus on relevant past frames
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple
import sys
import time
from scipy.signal import resample_poly

# Try Numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# All imports from local core (self-contained)
import os
_CORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core')
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

from crnn_attention import CRNN_GRU_Attention
from tiny_gru_alpha import extract_features
# from tiny_cnn_16k_native import TinyCNN16kNative, extract_features_16k_native
from frequency_band_omlsa import FrequencyBandOMLSA16k


# Numba-accelerated gain computation loop
if HAS_NUMBA:
    @jit(nopython=True, cache=True, nogil=True)
    def apply_active_forensic_gain_numba(
        spectrogram, power_spectrum, noise_est, 
        smoothed_spp, alpha_seq, adaptive_floors,
        trace_L_seq, trace_C_seq,  # Active Forensic Witnesses
        band_oversubtract,
        n_bins, low_start, mid_end,
        spp_gate, hf_start_bin,
        # Forensic params (from LearnedSelector)
        l_threshold, c_threshold, spp_power, smoothing
    ):
        """
        Active Forensic Gain Loop with L penalty and C gate.
        
        - trace_L_seq: Energy Flux per frame (L proxy) - high = suspicious
        - trace_C_seq: Spectral Coherence per frame (C proxy) - low = suspicious
        """
        n_frames = len(smoothed_spp)
        enhanced = np.zeros_like(spectrogram)
        prev_clean_sq = power_spectrum[0].copy()
        prev_forensic_mask = 0.5  # Initialize for smoothing
        
        for i in range(n_frames):
            frame_spp = smoothed_spp[i]
            frame_alpha = alpha_seq[i]
            
            noisy_power = power_spectrum[i]
            
            # ============ ACTIVE FORENSIC SUPPRESSION ============
            # L Penalty: Use param threshold
            l_penalty = 1.0
            if trace_L_seq[i] > l_threshold:
                # Moderate penalty for suspicious frame
                l_penalty = max(0.05, np.exp(-(trace_L_seq[i] - l_threshold) * 4.0))
            
            # C Gate: Use param threshold
            c_gate = 1.0
            if trace_C_seq[i] < c_threshold:
                c_gate = max(0.1, (trace_C_seq[i] / c_threshold) ** 1.5)
            
            # Combined forensic mask with SPP power param
            raw_forensic_mask = (frame_spp ** spp_power) * l_penalty * c_gate
            raw_forensic_mask = max(0.0, min(raw_forensic_mask, 1.0))
            
            # ============ TEMPORAL SMOOTHING ============
            # ============ TEMPORAL SMOOTHING ============
            forensic_mask = smoothing * prev_forensic_mask + (1.0 - smoothing) * raw_forensic_mask
            prev_forensic_mask = forensic_mask

            # ============ DUAL-MODE MASKING (User Request) ============
            # Ensure silence is brutally silent, but speech is protected
            if frame_spp < 0.5:
                 # SILENCE MODE: Aggressive non-linear suppression
                 # Simulates the "Old Formula" effect but only on noise
                 # If mask=0.3 -> 0.3^3 = 0.027 (Silenced!)
                 forensic_mask = forensic_mask ** 3.0
            else:
                 # SPEECH MODE: Gentle linear or boost
                 # If mask=0.8 -> 0.8 (Linear protection)
                 # We trust the relaxed L/C params here
                 forensic_mask = np.sqrt(forensic_mask) # Boost confidence (0.8->0.9)
            
            # ============ GAIN COMPUTATION ============
            # Get oversubtraction for this SPP level
            oversub_idx = min(int(frame_spp * 10), 9)
            oversub = band_oversubtract[oversub_idx]
            noise_adj = noise_est[i] * oversub
            
            # SNR estimation
            gamma_snr = noisy_power / np.maximum(noise_adj, 1e-10)
            
            # Decision-directed prior SNR
            xi_prev = prev_clean_sq / np.maximum(noise_est[i], 1e-10)
            xi_ml = np.maximum(gamma_snr - 1, 0)
            xi = frame_alpha * xi_prev + (1 - frame_alpha) * xi_ml
            xi = np.maximum(xi, 1e-3)
            
            # Wiener gain * forensic mask (HARD MASK for true silence)
            gain = (xi / (1.0 + xi)) * forensic_mask
            gain = np.clip(gain, 0.0, 1.0)
            
            # Apply floor to speech frequencies (with relaxed floor condition)
            floor = adaptive_floors[i] if forensic_mask > 0.3 else 0.005 # Increased from 0.0001 but lower than 0.02
            for j in range(low_start, mid_end):
                if gain[j] < floor:
                    gain[j] = floor
                    
            # SILENCE GATE / EXPANDER (RELAXED)
            if frame_spp < spp_gate:
                gate_factor = max(0.05, (frame_spp / spp_gate) ** 3)
                gain *= gate_factor
            
            # ============ STATIONARY NOISE GATE (RELAXED) ============
            frame_energy = np.mean(noisy_power)
            
            # ABSOLUTE SILENCE (Kill Switch)
            # If confidence is low (< 20%), force digital black (0.0)
            # This ensures clean training data for AI
            if frame_spp < 0.20:
                gain *= 0.0  # Absolute zero
            
            # Additional safety: If Energy is super low, force zero too
            if frame_energy < 1e-8: # ~-80dB
                gain *= 0.0
            
            # SPEECH PUNCH BOOST (Make speech clearer)
            # If model is confident it's speech, ensure it punches through
            if frame_spp > 0.8:
                gain *= 1.2  # Boost by 20%
                gain = np.clip(gain, 0.0, 1.0)
            
            energy_threshold = 1e-7
            if frame_energy < energy_threshold * 10:
                energy_factor = max(0.05, frame_energy / (energy_threshold * 10))
                gain *= energy_factor
            
            # SMART HF DAMPING (Refined to prevent squeezing)
            # Only damp HF if we are VERY sure it's noise/forensic-fail
            if forensic_mask < 0.2:  # Was 0.5 - allow more HF through unless very suspicious
                gain[hf_start_bin:] *= 0.1  # Was 0.05 - less aggressive cut
            elif frame_spp > 0.4:
                # Preserve sibilants when speech is detected
                pass
            else:
                # Gradual HF reduction
                hf_factor = frame_spp + 0.4  # Slightly more permissive
                if hf_factor > 1.0: hf_factor = 1.0
                gain[hf_start_bin:] *= hf_factor
            
            enhanced[i] = gain * spectrogram[i]
            prev_clean_sq = np.abs(enhanced[i]) ** 2
        
        return enhanced


    # Numba-accelerated overlap-add for iSTFT
    @jit(nopython=True, cache=True, nogil=True)
    def istft_overlap_add_numba(frames, window, hop_size, output_len):
        """JIT-compiled overlap-add."""
        n_frames = frames.shape[0]
        frame_size = frames.shape[1]
        
        output = np.zeros(output_len)
        window_sum = np.zeros(output_len)
        window_sq = window ** 2
        
        for i in range(n_frames):
            start = i * hop_size
            for j in range(frame_size):
                if start + j < output_len:
                    output[start + j] += frames[i, j] * window[j]
                    window_sum[start + j] += window_sq[j]
        
        # Normalize
        for i in range(output_len):
            if window_sum[i] > 1e-8:
                output[i] /= window_sum[i]
        return output


class AudioProcessorHybrid16kNative:
    """
    Fast Hybrid 16kHz processor with pre-computed SPP.
    
    Optimized for Speed:
    - Native 16kHz processing (no resample)
    - Frame Size 512 (32ms)
    - Hop Size 256 (16ms) -> 50% overlap (Standard is 75%, this is 2x faster)
    - Direct feature extraction
    """
    
    def __init__(
        self,
        model_path: str = "core/crnn_attention.pth",
        spp_smooth: float = 0.25,
        spp_bias: float = 0.10,           # Increased slightly more (punchy speech)
        spp_ceiling: float = 0.98,        # Allow almost full gain
        vad_threshold: float = 0.40,
        gain_floor_min: float = 0.02,     # Lowered back to 0.02 (cleaner silence) but safe
        suppression_scale: float = 6.0,   # Balanced (was 5.0, tighter for silence)
        spp_gate: float = 0.35,           # Tightened gate (was 0.30)
        hf_damp_freq: int = 4500,
        # Forensic parameters (Balanced)
        l_threshold: float = 2.0,         # Balanced
        c_threshold: float = 0.25,        # Balanced
        spp_power: float = 1.05,          # Slight power curve
        smoothing: float = 0.9,
        l_scaling: float = 0.10,
    ):
        self.sample_rate = 16000
        self.frame_size = 512
        self.hop_size = 256
        self.n_fft = 512
        self.n_bins = self.n_fft // 2 + 1
        
        self.spp_smooth = spp_smooth
        self.spp_bias = spp_bias
        self.spp_ceiling = spp_ceiling
        self.vad_threshold = vad_threshold
        self.gain_floor_min = gain_floor_min
        self.suppression_scale = suppression_scale
        self.spp_gate = spp_gate
        self.hf_damp_freq = hf_damp_freq
        
        # Forensic params
        self.l_threshold = l_threshold
        self.c_threshold = c_threshold
        self.spp_power = spp_power
        self.smoothing = smoothing
        self.l_scaling = l_scaling
        
        self.window = np.hanning(self.frame_size)
        
        # Load CRNN_GRU_Attention model (ultimate quality)
        self.model = None
        try:
            self.model = CRNN_GRU_Attention(input_size=5, cnn_channels=48, gru_hidden=64, gru_layers=2, num_heads=4)
            if Path(model_path).exists():
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"[OK] Loaded CRNN_GRU_Attention (99K params): {model_path}")
            else:
                print(f"[WARN] Model not found: {model_path}")
        except Exception as e:
            print(f"[ERROR] Loading model: {e}")
        
        # Frequency band processor for 16kHz
        self.freq_band = FrequencyBandOMLSA16k(n_bins=self.n_bins, sample_rate=16000)
        
        print(f"[OK] CRNN+GRU+Attention 16kHz Processor (Ultimate Quality)")
        print(f"     Frame: {self.frame_size}, Hop: {self.hop_size} (50% Overlap)")
    
    def stft(self, audio: np.ndarray) -> np.ndarray:
        """Vectorized STFT for entire audio."""
        n_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        
        # Create frame indices: shape (n_frames, frame_size)
        indices = np.arange(self.frame_size)[None, :] + \
                  np.arange(n_frames)[:, None] * self.hop_size
        
        # Batch extract frames
        frames = audio[indices] * self.window
        
        # Batch FFT
        spectrogram = np.fft.rfft(frames, self.n_fft)
        
        return spectrogram
    
    def istft(self, spectrogram: np.ndarray) -> np.ndarray:
        """Overlap-add iSTFT with Numba acceleration."""
        n_frames = spectrogram.shape[0]
        output_len = (n_frames - 1) * self.hop_size + self.frame_size
        
        # Batch iFFT
        frames = np.fft.irfft(spectrogram, self.n_fft)
        
        if HAS_NUMBA:
            # Use Numba for overlap-add
            return istft_overlap_add_numba(
                frames.astype(np.float64), 
                self.window.astype(np.float64), 
                self.hop_size, 
                output_len
            )
        else:
            # Fallback to Python
            output = np.zeros(output_len)
            window_sum = np.zeros(output_len)
            
            for i in range(n_frames):
                start = i * self.hop_size
                frame = frames[i] * self.window
                output[start:start + self.frame_size] += frame
                window_sum[start:start + self.frame_size] += self.window ** 2
            
            window_sum = np.maximum(window_sum, 1e-8)
            return output / window_sum
    
    def predict_spp_batch(self, audio_16k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict SPP using 8k model (Hybrid Control).
        Downsamples 16k -> 8k for VAD, keeping 16k for Audio.
        """
        if self.model is None:
            n_frames = (len(audio_16k) - self.frame_size) // self.hop_size + 1
            return np.full(n_frames, 0.98), np.full(n_frames, 0.5)
        
        # 1. Downsample to 8k for Model Input
        # 16k samples -> 8k samples (factor 0.5)
        # Use resample_poly for speed (decimate by 2)
        audio_8k = resample_poly(audio_16k, 1, 2)
        
        # 2. Extract features at 8k
        # 16k parameters: hop=256 (16ms)
        # 8k parameters: hop=128 (16ms)
        # Frame alignment is preserved (time-wise)
        features = extract_features(
            audio_8k, 
            sample_rate=8000, 
            hop_size=128 # 16ms hop at 8k
        )
        
        # Model inference - CHUNKED to avoid OOM with Attention
        # Attention is O(n²) so we need to process in chunks
        CHUNK_SIZE = 500  # ~4 seconds at 8kHz
        OVERLAP = 50      # Overlap for continuity
        
        n_frames = features.shape[0]
        all_alpha = []
        all_spp = []
        all_latent = []  # Collect latent for forensics
        hidden = None
        
        with torch.no_grad():
            for start in range(0, n_frames, CHUNK_SIZE - OVERLAP):
                end = min(start + CHUNK_SIZE, n_frames)
                chunk = features[start:end]
                
                x = torch.FloatTensor(chunk).unsqueeze(0)
                out, latent, hidden = self.model(x, hidden)  # Get latent
                out = out.squeeze(0).numpy()
                latent_np = latent.squeeze(0).numpy()  # (chunk_len, 64)
                
                # Remove overlap from previous chunk (except first)
                if start > 0:
                    out = out[OVERLAP:]
                    latent_np = latent_np[OVERLAP:]
                
                all_alpha.append(out[:, 0])
                all_spp.append(out[:, 1])
                all_latent.append(latent_np)
        
        alpha = np.concatenate(all_alpha)
        spp = np.concatenate(all_spp)
        latent = np.concatenate(all_latent, axis=0)  # (n_frames, 64)
        
        # Boost SPP by 20% to compensate for model predicting lower values
        spp = spp * 1.2
        
        return np.clip(alpha, 0.92, 0.995), np.clip(spp, 0, 1), latent
    
    def smooth_spp(self, raw_spp: np.ndarray) -> np.ndarray:
        """Apply asymmetric SPP smoothing."""
        n_frames = len(raw_spp)
        smoothed = np.zeros(n_frames)
        
        spp_attack = 0.4
        spp_decay = 0.45  # Rapid release (was 0.4)
        prev_spp = 0.5
        
        for i in range(n_frames):
            biased = raw_spp[i] + self.spp_bias
            if biased > prev_spp:
                smoothed[i] = spp_attack * prev_spp + (1 - spp_attack) * biased
            else:
                smoothed[i] = spp_decay * prev_spp + (1 - spp_decay) * biased
            prev_spp = smoothed[i]
        
        return np.clip(smoothed, 0.0, self.spp_ceiling)
    
    def estimate_noise_batch(self, power_spectrum: np.ndarray, spp: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Estimate noise for all frames."""
        n_frames, n_bins = power_spectrum.shape
        noise_est = np.zeros_like(power_spectrum)
        
        # Initialize with first frame
        noise_est[0] = power_spectrum[0]
        
        for i in range(1, n_frames):
            frame_spp = spp[i]
            
            # Hard VAD gating
            if frame_spp < self.vad_threshold:
                alpha_d = 0.85
            else:
                alpha_d = 0.98 + frame_spp * 0.015
            
            alpha_d = min(alpha_d, 0.995)
            noise_est[i] = alpha_d * noise_est[i-1] + (1 - alpha_d) * power_spectrum[i]
        
        return noise_est
    
    def compute_gain(self, xi: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """Vectorized OM-LSA gain computation."""
        xi_safe = np.maximum(xi, 1e-3)
        gamma_safe = np.maximum(gamma, 1e-3)
        
        v = xi_safe * gamma_safe / (1 + xi_safe)
        
        # Approximate special functions for speed
        with np.errstate(divide='ignore', invalid='ignore'):
            exp_v = np.exp(-v)
            gain = xi_safe / (1 + xi_safe) * np.sqrt(v / gamma_safe) * np.where(
                v < 10,
                (1 + v) * exp_v + v * (1 - exp_v),
                np.sqrt(v)
            )
        
        gain = np.nan_to_num(gain, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(gain, 0.0, 1.0)
    
    def process(self, audio: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, dict]:
        """Process entire audio with pre-computed SPP."""
        start_time = time.perf_counter()
        
        # Resample input to 16k if needed
        # Resample input to 16k if needed
        if sr != 16000:
            # Calculate GCD for optimal resampling
            import math
            gcd = math.gcd(16000, sr)
            audio = resample_poly(audio, 16000 // gcd, sr // gcd)
            sr = 16000
        
        original_length = len(audio)
        
        # Normalize
        max_val = np.max(np.abs(audio)) + 1e-10
        audio_norm = audio / max_val
        
        # ============================================
        # Phase 1: Pre-compute (sequential)
        # ============================================
        print("  [1/3] Pre-computing STFT and SPP (16kHz Native)...")
        phase1_start = time.perf_counter()
        
        # STFT
        spectrogram = self.stft(audio_norm)
        n_frames = spectrogram.shape[0]
        power_spectrum = np.abs(spectrogram) ** 2
        
        # SPP prediction (with latent for forensics)
        alpha_seq, spp_seq, latent_np = self.predict_spp_batch(audio_norm)
        
        # Ensure same length
        min_len = min(n_frames, len(spp_seq))
        spp_seq = spp_seq[:min_len]
        alpha_seq = alpha_seq[:min_len]
        latent_np = latent_np[:min_len]  # (min_len, 64)
        
        # Smooth SPP
        smoothed_spp = self.smooth_spp(spp_seq)
        
        # Adaptive floors
        adaptive_floors = self.gain_floor_min + (smoothed_spp * 0.22)
        adaptive_floors = np.minimum(adaptive_floors, 0.45)
        
        # Noise estimation
        noise_est = self.estimate_noise_batch(power_spectrum[:min_len], smoothed_spp, alpha_seq)
        
        phase1_time = time.perf_counter() - phase1_start
        print(f"  [1/3] Done in {phase1_time:.1f}s")
        
        # ============================================
        # Phase 2: Apply gain (Numba JIT if available)
        # ============================================
        print("  [2/3] Applying gain...")
        phase2_start = time.perf_counter()
        
        n_bins = self.n_bins
        # Low/Mid bands adjusted for 16k bins (Nyquist 8000)
        # 0-5.5kHz protected area
        low_start = int(n_bins * 200 / 8000)
        mid_end = int(n_bins * 5500 / 8000)
        
        if HAS_NUMBA:
            # Prepare lookup tables for Numba
            band_oversubtract = np.ones((10, n_bins), dtype=np.float64)
            
            for spp_idx in range(10):
                spp_val = spp_idx / 10.0
                # Apply global suppression scale
                oversub = self.freq_band.get_band_oversubtract(spp_val) * self.suppression_scale
                band_oversubtract[spp_idx] = oversub
            
            # Calculate HF bin start
            # 257 bins = 0-8000Hz. 4000Hz is roughly bin 128
            hf_start_bin = int((self.hf_damp_freq / (self.sample_rate / 2)) * self.n_bins)
            
            # ============================================
            # ACTIVE FORENSIC: Compute Trace L from GRU Latent (TRUE!)
            # ============================================
            # Trace L: Latent trajectory (frame-to-frame change in GRU hidden space)
            latent_norm = np.linalg.norm(latent_np, axis=1)  # Per-frame magnitude
            trace_L_raw = np.abs(np.diff(latent_norm, prepend=latent_norm[0]))
            
            # ABSOLUTE SCALING: Use param from LearnedSelector
            trace_L_raw = trace_L_raw / self.l_scaling  # Adaptive, not hard-coded!
            
            # Smooth L with exponential moving average (prevent sudden gating)
            trace_L_seq = np.zeros(min_len, dtype=np.float64)
            trace_L_seq[0] = trace_L_raw[0]
            for i in range(1, min_len):
                trace_L_seq[i] = 0.7 * trace_L_seq[i-1] + 0.3 * trace_L_raw[i]
            
            print(f"  [2/3] Trace L from GRU Latent (True Forensic)")
            
            # Trace C: Spectral Coherence (frame-to-frame correlation)
            trace_C_raw = np.ones(min_len, dtype=np.float64) * 0.8
            for i in range(1, min_len):
                corr = np.corrcoef(np.abs(spectrogram[i-1]), np.abs(spectrogram[i]))[0, 1]
                if not np.isnan(corr):
                    trace_C_raw[i] = max(0.0, corr)
            
            # Smooth C with exponential moving average
            trace_C_seq = np.zeros(min_len, dtype=np.float64)
            trace_C_seq[0] = trace_C_raw[0]
            for i in range(1, min_len):
                trace_C_seq[i] = 0.7 * trace_C_seq[i-1] + 0.3 * trace_C_raw[i]
            
            enhanced_spectrogram = apply_active_forensic_gain_numba(
                spectrogram[:min_len].astype(np.complex128),
                power_spectrum[:min_len].astype(np.float64),
                noise_est.astype(np.float64),
                smoothed_spp.astype(np.float64),
                alpha_seq.astype(np.float64),
                adaptive_floors.astype(np.float64),
                trace_L_seq.astype(np.float64),
                trace_C_seq.astype(np.float64),
                band_oversubtract,
                n_bins, low_start, mid_end,
                self.spp_gate,
                hf_start_bin,
                # Forensic params
                self.l_threshold,
                self.c_threshold,
                self.spp_power,
                self.smoothing
            )
            print("  [2/3] (Numba JIT - Active Forensic Gain)")
        else:
            # Fallback to Python loop
            enhanced_spectrogram = np.zeros_like(spectrogram[:min_len])
            prev_clean_sq = power_spectrum[0]
            
            for i in range(min_len):
                frame_spp = smoothed_spp[i]
                frame_alpha = alpha_seq[i]
                
                noisy_power = power_spectrum[i]
                noisy_spec = spectrogram[i]
                
                # Oversubtraction
                oversub = self.freq_band.get_band_oversubtract(frame_spp) * self.suppression_scale
                noise_adj = noise_est[i] * oversub
                
                # SNR estimation
                gamma_snr = noisy_power / np.maximum(noise_adj, 1e-10)
                
                # Decision-directed prior SNR
                xi_prev = prev_clean_sq / np.maximum(noise_est[i], 1e-10)
                xi_ml = np.maximum(gamma_snr - 1, 0)
                xi = frame_alpha * xi_prev + (1 - frame_alpha) * xi_ml
                xi = np.maximum(xi, 1e-3)
                
                # Compute gain
                base_gain = self.compute_gain(xi, gamma_snr)
                
                # Apply band floors ONLY if speech is present (SPP > gate)
                # This prevents the band floors (0.08 etc) from keeping noise alive in silence
                if frame_spp > self.spp_gate:
                     gain = self.freq_band.apply_gain(base_gain, frame_spp)
                else:
                     gain = base_gain # Use raw Wiener gain which can go very low
                
                # Apply floor to speech frequencies (global floor)
                gain[low_start:mid_end] = np.maximum(gain[low_start:mid_end], adaptive_floors[i])
                
                # Silence Gate / Expander
                if frame_spp < self.spp_gate:
                    gate_factor = (frame_spp / self.spp_gate) ** 4
                    gain *= gate_factor

                # Smart HF Damping (Python fallback)
                hf_start_bin = int((self.hf_damp_freq / (self.sample_rate / 2)) * self.n_bins)
                
                # We need raw power here, but we only have noisy_spec and gain
                # Re-calculate power for this check (approximate)
                current_power = np.abs(noisy_spec) ** 2
                hf_energy = np.mean(current_power[hf_start_bin:])
                low_start_idx = int(self.n_bins * 200 / 8000)
                mid_end_idx = int(self.n_bins * 5500 / 8000)
                lf_energy = np.mean(current_power[low_start_idx:mid_end_idx])
                
                hf_ratio = hf_energy / (lf_energy + 1e-10)
                
                if hf_ratio > 2.0 and frame_spp < 0.80:
                    hf_factor = 0.05
                elif frame_spp > 0.75:
                    hf_factor = 1.0
                elif hf_ratio > 1.5 and frame_spp < 0.70:
                    hf_factor = 0.3
                else:
                    hf_factor = (frame_spp ** 2) + 0.2
                    if hf_factor > 1.0: hf_factor = 1.0
                    
                gain[hf_start_bin:] *= hf_factor

                enhanced_spectrogram[i] = gain * noisy_spec
                prev_clean_sq = np.abs(enhanced_spectrogram[i]) ** 2
        
        phase2_time = time.perf_counter() - phase2_start
        print(f"  [2/3] Done in {phase2_time:.1f}s")
        
        # ============================================
        # Phase 3: Reconstruct
        # ============================================
        print("  [3/3] Reconstructing audio...")
        phase3_start = time.perf_counter()
        
        enhanced = self.istft(enhanced_spectrogram) * max_val
        
        # Soft limiter
        threshold = 0.95
        enhanced = np.where(
            np.abs(enhanced) > threshold,
            np.sign(enhanced) * (threshold + np.tanh((np.abs(enhanced) - threshold) * 2) * (1 - threshold)),
            enhanced
        )
        
        # Loudness compensation (RMS-based)
        # Relaxed clip to allow recovering volume after heavy noise reduction
        rms_in = np.sqrt(np.mean(audio ** 2) + 1e-10)
        rms_out = np.sqrt(np.mean(enhanced ** 2) + 1e-10)
        makeup = rms_in / rms_out
        makeup = np.clip(makeup, 0.8, 1.5)  # Reduced from 5.0 to prevent bloating
        enhanced = enhanced * makeup
        
        # Safe limiter
        max_val_out = np.max(np.abs(enhanced))
        if max_val_out > 0.95:
            enhanced = enhanced * (0.95 / max_val_out)
        
        phase3_time = time.perf_counter() - phase3_start
        print(f"  [3/3] Done in {phase3_time:.1f}s")
        
        # ============================================
        # Phase 4: Time-Domain Silence Gate (V4 Port)
        # ============================================
        # Final cleanup for absolute silence
        print("  [4/4] Applying Time-Domain Silence Gate...")
        enhanced = self._apply_silence_gate(enhanced, self.sample_rate)
        
        total_time = time.perf_counter() - start_time
        audio_duration = original_length / sr
        speed = audio_duration / total_time
        
        info = {
            'speed': speed,
            'total_time': total_time,
            'phase1_time': phase1_time,
            'phase2_time': phase2_time,
            'phase3_time': phase3_time,
            'n_frames': min_len,
        }
        
        return enhanced[:original_length], info

    def _apply_silence_gate(self, data: np.ndarray, sr: int, threshold_db: float = -35.0) -> np.ndarray:
        """
        Apply SMART Noise Gate with Attack + Release.
        - Attack: Gate opens after 50ms of continuous above-threshold (prevents pre-speech noise)
        - Release: Gate closes after 150ms of below-threshold (preserves tails)
        """
        # Calculate global peak
        global_peak = np.max(np.abs(data))
        if global_peak < 1e-4: return data
        
        # Threshold relative to peak
        rel_threshold_linear = global_peak * (10 ** (threshold_db / 20))
        
        # Attack time (50ms - gate opens slowly to reject pre-noise)
        attack_ms = 50
        attack_samples = int(sr * attack_ms / 1000)
        
        # Release time (200ms - preserve tails)
        release_ms = 200
        release_samples = int(sr * release_ms / 1000)
        
        # Window size (10ms)
        window_samples = int(sr * 0.01)
        if window_samples < 1: window_samples = 1
        
        n_windows = len(data) // window_samples
        if n_windows == 0: return data
        
        result = data.copy()
        print(f"       > Smart Gate: Attack={attack_ms}ms, Release={release_ms}ms, Thresh=-{abs(threshold_db):.0f}dB")
        
        # State tracking
        samples_above_threshold = 0  # For attack
        samples_below_threshold = release_samples + 1  # Start closed
        gate_open = False
        closed_count = 0
        
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            window = data[start:end]
            window_peak = np.max(np.abs(window))
            
            if window_peak > rel_threshold_linear:
                # Signal above threshold
                samples_above_threshold += window_samples
                samples_below_threshold = 0
                
                # ATTACK: Only open gate after sustained signal
                if samples_above_threshold >= attack_samples:
                    gate_open = True
            else:
                # Signal below threshold
                samples_below_threshold += window_samples
                samples_above_threshold = 0
                
                # RELEASE: Close gate after sustained silence
                if samples_below_threshold >= release_samples:
                    gate_open = False
            
            # Apply gate
            if not gate_open:
                result[start:end] = 0.0
                closed_count += 1

        # Handle leftover
        leftover_start = n_windows * window_samples
        if leftover_start < len(data):
            window = data[leftover_start:]
            if np.max(np.abs(window)) > rel_threshold_linear:
                samples_above_threshold += len(window)
                samples_below_threshold = 0
                if samples_above_threshold >= attack_samples:
                    gate_open = True
            else:
                samples_below_threshold += len(window)
                if samples_below_threshold >= release_samples:
                    gate_open = False
            
            if not gate_open:
                result[leftover_start:] = 0.0
        
        print(f"       > Gated {closed_count} windows to Digital Black")
        return result


def main():
    import argparse
    import soundfile as sf
    
    parser = argparse.ArgumentParser(description='Hybrid 16kHz Speech Enhancement (Native Fast)')
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('output', nargs='?', help='Output audio file')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    print("=" * 60)
    print("Hybrid 16kHz Native Speech Enhancement")
    print("=" * 60)
    
    # Load audio
    print(f"\n[Loading] {args.input}")
    audio, sr = sf.read(args.input)
    
    # Force resampling to 16k if not already
    if sr != 16000:
        print(f"  Resampling input from {sr}Hz to 16000Hz...")
        import math
        gcd = math.gcd(16000, sr)
        up = 16000 // gcd
        down = sr // gcd
        
        if audio.ndim == 2:
             audio_l = resample_poly(audio[:, 0], up, down)
             audio_r = resample_poly(audio[:, 1], up, down)
             audio = np.column_stack([audio_l, audio_r])
        else:
             audio = resample_poly(audio, up, down)
        sr = 16000
    
    # Handle stereo
    is_stereo = False
    was_stereo_silent = False
    
    left, right = None, None
    
    if audio.ndim == 2:
        left = audio[:, 0]
        right = audio[:, 1]
        is_stereo = True
        
        left_rms = np.sqrt(np.mean(left ** 2))
        right_rms = np.sqrt(np.mean(right ** 2))
        
        if right_rms < left_rms * 0.05:
            was_stereo_silent = True
            is_stereo = False
            left, right = left, None
        elif left_rms < right_rms * 0.05:
            was_stereo_silent = True
            is_stereo = False
            left, right = right, None
    else:
        left, right = audio, None
    
    mode_str = 'Stereo' if is_stereo else ('Dual Mono' if was_stereo_silent else 'Mono')
    print(f"  Duration: {len(left)/sr:.1f}s, SR: {sr}Hz, {mode_str}")
    
    total_start = time.perf_counter()
    
    # Use ThreadPool for stereo
    processor = AudioProcessorHybrid16kNative()
    
    print("\n[Enhancing...]")
    
    info_stats = {}
    
    # Parallel processing for Stereo
    if is_stereo:
        print("  Processing Stereo Channels in Parallel...")
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_l = executor.submit(processor.process, left, sr)
            future_r = executor.submit(processor.process, right, sr)
            
            enhanced_left, info_l = future_l.result()
            enhanced_right, info_r = future_r.result()
        
        info_stats = info_l
        
        min_len = min(len(enhanced_left), len(enhanced_right))
        enhanced = np.column_stack([enhanced_left[:min_len], enhanced_right[:min_len]])
            
    elif was_stereo_silent:
        print("  Processing Mono Channel...")
        enhanced_left, info = processor.process(left, sr)
        enhanced = np.column_stack([enhanced_left, enhanced_left])
        info_stats = info
    else:
        print("  Processing Mono Channel...")
        enhanced_left, info = processor.process(left, sr)
        enhanced = enhanced_left
        info_stats = info
    
    # Light normalization
    max_val = np.max(np.abs(enhanced))
    if max_val > 0.98:
        enhanced = enhanced * (0.98 / max_val)
    
    output_path = args.output or f"{input_path.stem}_native16k_hybrid.wav"
    sf.write(output_path, enhanced, 16000)
    
    total_elapsed = time.perf_counter() - total_start
    audio_duration = len(left) / sr
    total_speed = audio_duration / total_elapsed
    
    print("\n" + "=" * 60)
    print("[Complete]")
    print(f"  Output: {output_path}")
    print(f"  Output SR: 16kHz (Native)")
    print(f"  Time: {total_elapsed:.1f}s")
    print(f"  Speed: {total_speed:.1f}x realtime")
    print("=" * 60)


if __name__ == "__main__":
    main()
