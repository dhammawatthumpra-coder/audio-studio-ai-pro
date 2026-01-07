"""
Audio Processor V18 - Smart Minimal
====================================
Based on V17 insight: V7 x N passes for digital black, then blend + HF.

Pipeline:
1. Analyze source → decide 1 or 2 V7 passes
2. V7 x N → clean audio with digital black
3. Blend V7 output (silence) + original (speech) in SINGLE STFT
4. HF reduction in same STFT pass
5. Output

Total: V7 STFT + 1 final STFT = minimal
"""

import numpy as np
import time
from numba import jit


# ============================================
# NOISE FLOOR MEASUREMENT
# ============================================
def noise_floor_db(audio, sr=16000, percentile=5):
    """Measure noise floor in dB."""
    frame_size = int(0.03 * sr)
    hop_size = frame_size // 2
    n_frames = (len(audio) - frame_size) // hop_size
    if n_frames <= 0:
        return -60.0
    
    rms_values = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_size
        rms_values[i] = np.sqrt(np.mean(audio[start:start + frame_size] ** 2))
    
    noise_floor = np.percentile(rms_values, percentile)
    if noise_floor < 1e-10:
        return -80.0
    return 20 * np.log10(noise_floor)


# ============================================
# CREATE BLEND MASK FROM V7 OUTPUT
# ============================================
def create_blend_mask(v7_output, sr, threshold_ratio=0.005):
    """Simple mask: 1=speech (use original), 0=silence (use V7)"""
    window_samples = int(sr * 0.01)  # 10ms
    n_windows = len(v7_output) // window_samples
    
    global_peak = np.max(np.abs(v7_output))
    if global_peak < 1e-6:
        return np.zeros_like(v7_output)
    
    threshold = global_peak * threshold_ratio
    mask = np.zeros_like(v7_output)
    
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        if np.max(np.abs(v7_output[start:end])) > threshold:
            mask[start:end] = 1.0
    
    return mask


# ============================================
# V18 SMART PROCESSOR
# ============================================
class AudioProcessorV18Smart:
    """V18 Smart = Auto V7 passes + Single-STFT blend + HF"""
    
    def __init__(self, model_path: str = None):
        import os
        _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Use absolute path for model
        if model_path is None:
            model_path = os.path.join(_BASE_DIR, 'core', 'crnn_attention.pth')
        
        self.sample_rate = 16000
        self.n_fft = 512
        self.hop_size = 256
        self.n_bins = self.n_fft // 2 + 1
        self.hf_start_bin = 120  # ~3.7kHz
        
        # V7 processor
        from audio_processor_crnn_attention_latent import AudioProcessorHybrid16kNative
        self.v7_processor = AudioProcessorHybrid16kNative(model_path=model_path)
        
        self.window = np.hanning(self.n_fft)
        
        print("[OK] V18 Smart Processor Initialized")
        print("     = Auto V7 passes + Single-STFT Blend + HF")
    
    def stft(self, audio):
        n_frames = (len(audio) - self.n_fft) // self.hop_size + 1
        spec = np.zeros((n_frames, self.n_bins), dtype=np.complex128)
        for i in range(n_frames):
            start = i * self.hop_size
            spec[i] = np.fft.rfft(audio[start:start+self.n_fft] * self.window, n=self.n_fft)
        return spec
    
    def istft(self, spec):
        n_frames = spec.shape[0]
        out_len = (n_frames - 1) * self.hop_size + self.n_fft
        out = np.zeros(out_len)
        win_sum = np.zeros(out_len)
        for i in range(n_frames):
            start = i * self.hop_size
            frame = np.real(np.fft.irfft(spec[i], n=self.n_fft))[:self.n_fft]
            out[start:start+self.n_fft] += frame * self.window
            win_sum[start:start+self.n_fft] += self.window ** 2
        win_sum = np.maximum(win_sum, 1e-8)
        return out / win_sum
    
    def process(self, audio: np.ndarray, sr: int = 16000, noise_threshold: float = -35.0):
        """Smart processing with auto V7 passes."""
        start_time = time.perf_counter()
        original_length = len(audio)
        
        max_val = np.max(np.abs(audio))
        if max_val < 1e-6:
            max_val = 1.0
        audio_norm = audio / max_val
        
        # Analyze source
        initial_nf = noise_floor_db(audio_norm, sr)
        needs_2pass = initial_nf > noise_threshold
        n_passes = 2 if needs_2pass else 1
        print(f"  [ANALYZE] Noise: {initial_nf:.1f} dB -> {n_passes} pass(es)")
        
        # V7 Pass 1: Initial cleaning (save for voice quality)
        print("  [V7] Pass 1 - Initial cleaning...")
        clean1, _ = self.v7_processor.process(audio_norm, sr)
        
        # V7 Pass 2: Better digital black (only if noisy)
        if needs_2pass:
            print("  [V7] Pass 2 - Digital Black refinement...")
            clean2, _ = self.v7_processor.process(clean1, sr)
        else:
            clean2 = clean1
        
        # Match lengths
        min_len = min(len(clean1), len(clean2), original_length)
        clean1 = clean1[:min_len]
        clean2 = clean2[:min_len]
        
        # Pad if needed
        if len(clean1) < original_length:
            clean1 = np.pad(clean1, (0, original_length - len(clean1)))
            clean2 = np.pad(clean2, (0, original_length - len(clean2)))
        
        # Create blend mask from Pass 2 (better silence detection)
        print("  [BLEND] Creating mask from Pass 2...")
        blend_mask = create_blend_mask(clean2, sr)
        speech_pct = np.mean(blend_mask) * 100
        print(f"          Speech: {speech_pct:.1f}%")
        
        # SINGLE STFT: Blend clean1 (speech) + clean2 (silence) + HF
        print("  [FINAL] Single-STFT Blend + HF...")
        clean1_spec = self.stft(clean1)
        clean2_spec = self.stft(clean2)
        
        n_frames = min(clean1_spec.shape[0], clean2_spec.shape[0])
        final_spec = np.zeros((n_frames, self.n_bins), dtype=np.complex128)
        
        for i in range(n_frames):
            # Get frame-level mask
            start = i * self.hop_size
            end = min(start + self.hop_size, len(blend_mask))
            frame_mask = np.mean(blend_mask[start:end]) if end > start else 0
            
            # Blend: speech=clean1 (better voice), silence=clean2 (better digital black)
            blended = frame_mask * clean1_spec[i] + (1 - frame_mask) * clean2_spec[i]
            
            # HF reduction in speech regions only
            if frame_mask > 0.5:
                for j in range(self.hf_start_bin, self.n_bins):
                    freq_ratio = (j - self.hf_start_bin) / max(1, (self.n_bins - self.hf_start_bin))
                    reduction = 0.3 - (freq_ratio * 0.1)  # Keep 30-20%
                    reduction = max(0.2, min(1.0, reduction))
                    blended[j] *= reduction
            
            final_spec[i] = blended
        
        enhanced = self.istft(final_spec) * max_val
        
        # Match length
        if len(enhanced) > original_length:
            enhanced = enhanced[:original_length]
        elif len(enhanced) < original_length:
            enhanced = np.pad(enhanced, (0, original_length - len(enhanced)))
        
        # Loudness compensation
        orig_rms = np.sqrt(np.mean(audio ** 2))
        enh_rms = np.sqrt(np.mean(enhanced ** 2))
        if enh_rms > 1e-10:
            factor = np.clip(orig_rms / enh_rms, 0.8, 1.5)
            enhanced *= factor
            print(f"  [LOUDNESS] {factor:.2f}x")
        
        # Minimum loudness floor (prevent too quiet)
        min_rms = 0.05  # Target minimum RMS
        current_rms = np.sqrt(np.mean(enhanced ** 2))
        if current_rms < min_rms and current_rms > 1e-10:
            boost = min_rms / current_rms
            boost = min(boost, 3.0)  # Max 3x boost
            enhanced *= boost
            print(f"  [BOOST] Quiet source: {boost:.2f}x")
        
        # Soft limiter (gentler)
        peak = np.max(np.abs(enhanced))
        if peak > 0.98:
            enhanced *= 0.98 / peak
            print(f"  [LIMITER] {peak:.2f} -> 0.98")
        
        elapsed = time.perf_counter() - start_time
        
        return enhanced, {
            'speed': (original_length / sr) / elapsed,
            'speech_ratio': speech_pct / 100,
            'passes': n_passes
        }
