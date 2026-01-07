"""
Audio Processor Hybrid 8kHz - Fast Version
==========================================
8kHz Native version of the hybrid processor.
Optimized for maximum speed on 8kHz content (telephony, crude recordings).

Speed improvements vs 16kHz Fast:
- 1/2 FFT size
- No resampling overhead
- Direct features (no downsampling)
- Extreme Speed Mode: 16ms Hop Size (50% overlap)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple
import sys
import os
import time
from scipy.signal import resample
from scipy.special import expn

# Try Numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# All imports from local core (self-contained)
_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_current_dir, 'core'))

from tiny_cnn_v2 import TinyCNNV2
from tiny_gru_alpha import extract_features

# Band configuration for 8kHz (Nyquist = 4kHz)
BAND_CONFIGS_8K = {
    "low": {
        "range": (0, 300),
        "oversubtract": 2.5,
        "floor": 0.08,
    },
    "mid_low": {
        "range": (300, 1200),
        "oversubtract": 1.6,
        "floor": 0.15,
    },
    "mid": {
        "range": (1200, 2800),
        "oversubtract": 1.0,  # Protect speech primary vocal range
        "floor": 0.18,
    },
    "high": {
        "range": (2800, 4000),
        "oversubtract": 3.0,  # Suppress upper hiss
        "floor": 0.05,
    },
}


class FrequencyBandOMLSA8k:
    """Frequency-band specific processing for 8kHz."""
    
    def __init__(self, n_bins: int = 129, sample_rate: int = 8000):
        self.n_bins = n_bins
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        
        # Create band masks
        self.band_masks = {}
        self.band_oversub = {}
        self.band_floors = {}
        
        for band_name, config in BAND_CONFIGS_8K.items():
            low_hz, high_hz = config["range"]
            # Map Hz to bins
            low_bin = int(low_hz / self.nyquist * (n_bins - 1))
            high_bin = int(high_hz / self.nyquist * (n_bins - 1))
            
            # Ensure valid range
            low_bin = max(0, low_bin)
            high_bin = min(n_bins, high_bin)
            
            mask = np.zeros(n_bins, dtype=bool)
            mask[low_bin:high_bin] = True
            
            self.band_masks[band_name] = mask
            self.band_oversub[band_name] = config["oversubtract"]
            self.band_floors[band_name] = config["floor"]
    
    def get_band_oversubtract(self, spp: float) -> np.ndarray:
        """Get per-bin oversubtraction factors."""
        oversub = np.ones(self.n_bins)
        
        # SPP-dependent suppression
        speech_protection = 1.0 - (spp * 0.4)
        
        for band_name, mask in self.band_masks.items():
            base = self.band_oversub[band_name]
            
            if band_name == "high":
                # High band: only protect during CLEAR speech
                if spp > 0.7: 
                    oversub[mask] = base * 0.6  
                else:
                    oversub[mask] = base
            else:
                # Lower bands: normal behavior
                oversub[mask] = base * speech_protection
        
        return oversub
    
    def apply_gain(self, gain: np.ndarray, spp: float) -> np.ndarray:
        """Apply band-specific gain floors."""
        result = gain.copy()
        
        for band_name, mask in self.band_masks.items():
            floor = self.band_floors[band_name]
            if band_name in ["mid_low", "mid"]:
                floor = floor + (spp * 0.08)
            result[mask] = np.maximum(result[mask], floor)
        
        return result


# Numba-accelerated gain computation loop
if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def apply_gain_numba(
        spectrogram, power_spectrum, noise_est, 
        smoothed_spp, alpha_seq, adaptive_floors,
        band_oversubtract,
        n_bins, low_start, mid_end
    ):
        """JIT-compiled gain loop with stable Wiener gain."""
        n_frames = len(smoothed_spp)
        enhanced = np.zeros_like(spectrogram)
        prev_clean_sq = power_spectrum[0].copy()
        
        for i in range(n_frames):
            frame_spp = smoothed_spp[i]
            frame_alpha = alpha_seq[i]
            
            noisy_power = power_spectrum[i]
            
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
            
            # Simple Wiener gain (stable, no decay)
            gain = xi / (1.0 + xi)
            gain = np.clip(gain, 0.02, 1.0)  # Minimum floor
            
            # Apply floor to speech frequencies
            for j in range(low_start, mid_end):
                if gain[j] < adaptive_floors[i]:
                    gain[j] = adaptive_floors[i]
            
            enhanced[i] = gain * spectrogram[i]
            prev_clean_sq = np.abs(enhanced[i]) ** 2
        
        return enhanced


    # Numba-accelerated overlap-add for iSTFT
    @jit(nopython=True, cache=True)
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


class AudioProcessorHybrid8k:
    """
    Fast Hybrid 8kHz processor with pre-computed SPP.
    
    Optimized for Speed:
    - Native 8kHz processing (no resample)
    - Smaller FFT (256 vs 512)
    - Direct feature extraction
    - Hop size 128 (16ms) for 50% overlap (2x faster than 75% overlap)
    """
    
    def __init__(
        self,
        model_path: str = None,  # Will be set to absolute path
        spp_smooth: float = 0.25,
        spp_bias: float = 0.04,
        spp_ceiling: float = 0.92,
        vad_threshold: float = 0.20,
        gain_floor_min: float = 0.06,
    ):
        # Set default model path to absolute path
        if model_path is None:
            model_path = os.path.join(_current_dir, 'core', 'tiny_cnn_v2.pth')
        
        self.sample_rate = 8000
        self.frame_size = 256
        self.hop_size = 128      # 16ms hop (50% overlap)
        self.n_fft = 256
        self.n_bins = self.n_fft // 2 + 1
        
        self.spp_smooth = spp_smooth
        self.spp_bias = spp_bias
        self.spp_ceiling = spp_ceiling
        self.vad_threshold = vad_threshold
        self.gain_floor_min = gain_floor_min
        
        self.window = np.hanning(self.frame_size)
        
        # Load PyTorch model
        self.model = None
        try:
            self.model = TinyCNNV2(hidden_size=16)
            if Path(model_path).exists():
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"[OK] Loaded TinyCNN V2 (PyTorch): {model_path}")
            else:
                print(f"[WARN] Model not found: {model_path}")
        except Exception as e:
            print(f"[ERROR] Loading model: {e}")
        
        # Frequency band processor for 8kHz
        self.freq_band = FrequencyBandOMLSA8k(n_bins=self.n_bins, sample_rate=8000)
        
        print(f"[OK] Hybrid 8kHz Fast Processor (Pre-compute SPP)")
    
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
    
    def predict_spp_batch(self, audio_8k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict SPP for entire audio at once."""
        if self.model is None:
            n_frames = (len(audio_8k) - self.frame_size) // self.hop_size + 1
            return np.full(n_frames, 0.98), np.full(n_frames, 0.5)
        
        # Extract features (Directly on 8k with sync hop size)
        features = extract_features(audio_8k, sample_rate=8000, hop_size=self.hop_size)
        
        # Model inference
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            out = self.model(x)
            out = out.squeeze(0).numpy()
        
        alpha_8k = out[:, 0]
        spp_8k = out[:, 1]
        
        return np.clip(alpha_8k, 0.92, 0.995), np.clip(spp_8k, 0, 1)
    
    def smooth_spp(self, raw_spp: np.ndarray) -> np.ndarray:
        """Apply asymmetric SPP smoothing."""
        n_frames = len(raw_spp)
        smoothed = np.zeros(n_frames)
        
        spp_attack = 0.4
        spp_decay = 0.2  # Faster decay (User request: was 0.05)
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
    
    def process(self, audio: np.ndarray, sr: int = 8000) -> Tuple[np.ndarray, dict]:
        """Process entire audio with pre-computed SPP."""
        start_time = time.perf_counter()
        
        # Keep original info for resampling back
        original_sr = sr
        original_length = len(audio)
        
        # Resample input to 8k if needed
        if sr != 8000:
            from scipy.signal import resample_poly
            import math
            gcd = math.gcd(8000, sr)
            audio = resample_poly(audio, 8000 // gcd, sr // gcd)
            sr = 8000
        
        # Normalize
        max_val = np.max(np.abs(audio)) + 1e-10
        audio_norm = audio / max_val
        
        # ============================================
        # Phase 1: Pre-compute (sequential)
        # ============================================
        print("  [1/3] Pre-computing STFT and SPP (8kHz)...")
        phase1_start = time.perf_counter()
        
        # STFT
        spectrogram = self.stft(audio_norm)
        n_frames = spectrogram.shape[0]
        power_spectrum = np.abs(spectrogram) ** 2
        
        # SPP prediction (no downsampling needed)
        alpha_seq, spp_seq = self.predict_spp_batch(audio_norm)
        
        # Ensure same length
        min_len = min(n_frames, len(spp_seq))
        spp_seq = spp_seq[:min_len]
        alpha_seq = alpha_seq[:min_len]
        
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
        # Low/Mid bands adjusted for 8k bins (Nyquist 4000)
        # 0-300Hz ~ 0-10 bins
        # 300-1200Hz ~ 10-38 bins
        low_start = int(n_bins * 200 / 4000)
        mid_end = int(n_bins * 2800 / 4000)
        
        if HAS_NUMBA:
            # Prepare lookup tables for Numba
            band_oversubtract = np.ones((10, n_bins), dtype=np.float64)
            
            for spp_idx in range(10):
                spp_val = spp_idx / 10.0
                band_oversubtract[spp_idx] = self.freq_band.get_band_oversubtract(spp_val)
            
            enhanced_spectrogram = apply_gain_numba(
                spectrogram[:min_len].astype(np.complex128),
                power_spectrum[:min_len].astype(np.float64),
                noise_est.astype(np.float64),
                smoothed_spp.astype(np.float64),
                alpha_seq.astype(np.float64),
                adaptive_floors.astype(np.float64),
                band_oversubtract,
                n_bins, low_start, mid_end
            )
            print("  [2/3] (Numba JIT - Wiener gain)")
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
                oversub = self.freq_band.get_band_oversubtract(frame_spp)
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
                gain = self.freq_band.apply_gain(base_gain, frame_spp)
                
                # Apply floor to speech frequencies
                gain[low_start:mid_end] = np.maximum(gain[low_start:mid_end], adaptive_floors[i])
                
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
        rms_in = np.sqrt(np.mean(audio ** 2) + 1e-10)
        rms_out = np.sqrt(np.mean(enhanced ** 2) + 1e-10)
        makeup = rms_in / rms_out
        makeup = np.clip(makeup, 0.8, 1.2)  # Safety clip
        enhanced = enhanced * makeup
        
        # Safe limiter
        max_val_out = np.max(np.abs(enhanced))
        if max_val_out > 0.95:
            enhanced = enhanced * (0.95 / max_val_out)
        
        phase3_time = time.perf_counter() - phase3_start
        print(f"  [3/3] Done in {phase3_time:.1f}s")
        
        total_time = time.perf_counter() - start_time
        # Resample back to original sample rate if we downsampled
        if original_sr != 8000:
            from scipy.signal import resample_poly
            import math
            gcd = math.gcd(original_sr, 8000)
            enhanced = resample_poly(enhanced, original_sr // gcd, 8000 // gcd)
            # Trim to exact original length
            enhanced = enhanced[:original_length]
        
        audio_duration = original_length / original_sr
        speed = audio_duration / total_time
        
        info = {
            'speed': speed,
            'total_time': total_time,
            'phase1_time': phase1_time,
            'phase2_time': phase2_time,
            'phase3_time': phase3_time,
            'n_frames': min_len,
        }
        
        return enhanced, info


def main():
    import argparse
    import soundfile as sf
    from concurrent.futures import ThreadPoolExecutor
    
    parser = argparse.ArgumentParser(description='Hybrid 8kHz Speech Enhancement (Fast)')
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('output', nargs='?', help='Output audio file')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    print("=" * 60)
    print("Hybrid 8kHz Speech Enhancement (Extreme Speed)")
    print("=" * 60)
    
    # Load audio
    print(f"\n[Loading] {args.input}")
    audio, sr = sf.read(args.input)
    
    # Force resampling to 8k if not already
    if sr != 8000:
        print(f"  Resampling input from {sr}Hz to 8000Hz...")
        if audio.ndim == 2:
             audio_l = resample(audio[:, 0], int(len(audio) * 8000 / sr))
             audio_r = resample(audio[:, 1], int(len(audio) * 8000 / sr))
             audio = np.column_stack([audio_l, audio_r])
        else:
             audio = resample(audio, int(len(audio) * 8000 / sr))
        sr = 8000
    
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
    processor = AudioProcessorHybrid8k()
    
    print("\n[Enhancing...]")
    
    info_stats = {}
    
    # Sequential processing (Faster due to GIL contention in parallel mode)
    if is_stereo:
        print("  Processing Left Channel...")
        enhanced_left, info_l = processor.process(left, sr)
        
        print("  Processing Right Channel...")
        enhanced_right, info_r = processor.process(right, sr)
        
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
    
    output_path = args.output or f"{input_path.stem}_hybrid8k_fast.wav"
    sf.write(output_path, enhanced, 8000)
    
    total_elapsed = time.perf_counter() - total_start
    audio_duration = len(left) / sr
    total_speed = audio_duration / total_elapsed
    
    print("\n" + "=" * 60)
    print("[Complete]")
    print(f"  Output: {output_path}")
    print(f"  Output SR: 8kHz (native)")
    print(f"  Time: {total_elapsed:.1f}s")
    print(f"  Speed: {total_speed:.1f}x realtime")
    print("=" * 60)


if __name__ == "__main__":
    main()
