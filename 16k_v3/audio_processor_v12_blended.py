"""
Audio Processor V12 - Blended Output
=====================================
รวมจุดแข็งของทั้งสองระบบ:

- Speech regions: ใช้ CRNN output (เสียงพูดดีกว่า)
- Silence regions: ใช้ V7 output (Digital Black - ลด noise ได้ดี)

Pipeline:
1. Run V7 → v7_output (has digital black)
2. Run CRNN on Original → crnn_output (has better speech)
3. Create mask from V7 output
4. Blend: output = mask × crnn + (1-mask) × v7
"""

import numpy as np
import time
from pathlib import Path
from scipy.signal import resample_poly


# ============================================
# CREATE BLEND MASK FROM V7 OUTPUT (with Fade In/Out)
# ============================================
def create_blend_mask(
    v7_output: np.ndarray,
    sr: int,
    window_ms: int = 10,
    threshold_ratio: float = 0.005,
    smoothing: float = 0.8,
    expand_ms: int = 50,
    fade_in_ms: int = 30,   # Fade in at speech start
    fade_out_ms: int = 50   # Fade out at speech end
) -> np.ndarray:
    """
    Create a smooth blend mask from V7 output with Fade In/Out.
    
    Where V7 has signal → mask = 1 (use CRNN)
    Where V7 is silent → mask = 0 (use V7's digital black)
    
    NEW: Fade In/Out at speech edges for natural transitions.
    """
    window_samples = int(sr * window_ms / 1000)
    expand_samples = int(sr * expand_ms / 1000)
    fade_in_samples = int(sr * fade_in_ms / 1000)
    fade_out_samples = int(sr * fade_out_ms / 1000)
    
    if window_samples < 1:
        window_samples = 1
    
    n_windows = len(v7_output) // window_samples
    if n_windows == 0:
        return np.ones_like(v7_output)
    
    global_peak = np.max(np.abs(v7_output))
    if global_peak < 1e-6:
        return np.zeros_like(v7_output)
    
    threshold = global_peak * threshold_ratio
    
    # Create per-window binary mask
    mask = np.zeros_like(v7_output)
    
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        window_peak = np.max(np.abs(v7_output[start:end]))
        mask[start:end] = 1.0 if window_peak > threshold else 0.0
    
    # Handle leftover
    leftover_start = n_windows * window_samples
    if leftover_start < len(v7_output):
        window_peak = np.max(np.abs(v7_output[leftover_start:]))
        mask[leftover_start:] = 1.0 if window_peak > threshold else 0.0
    
    # DILATION: Expand mask around speech regions
    if expand_samples > 0:
        dilated_mask = mask.copy()
        for i in range(len(mask)):
            if mask[i] > 0.5:
                start_expand = max(0, i - expand_samples)
                dilated_mask[start_expand:i] = 1.0
                end_expand = min(len(mask), i + expand_samples)
                dilated_mask[i:end_expand] = 1.0
        mask = dilated_mask
    
    # ============================================
    # FADE IN/OUT AT SPEECH EDGES
    # ============================================
    # Find edges (transitions from 0→1 and 1→0)
    if fade_in_samples > 0 or fade_out_samples > 0:
        edges_start = []  # 0→1 transitions (start of speech)
        edges_end = []    # 1→0 transitions (end of speech)
        
        for i in range(1, len(mask)):
            if mask[i-1] < 0.5 and mask[i] > 0.5:
                edges_start.append(i)
            elif mask[i-1] > 0.5 and mask[i] < 0.5:
                edges_end.append(i)
        
        # Apply fade-in ramps
        for edge in edges_start:
            for j in range(fade_in_samples):
                idx = edge + j
                if idx < len(mask):
                    # Ramp from 0 to 1 (cosine for smooth curve)
                    t = j / fade_in_samples
                    fade_val = 0.5 - 0.5 * np.cos(np.pi * t)
                    mask[idx] = min(mask[idx], fade_val)
        
        # Apply fade-out ramps
        for edge in edges_end:
            for j in range(fade_out_samples):
                idx = edge - fade_out_samples + j
                if 0 <= idx < len(mask):
                    # Ramp from 1 to 0 (cosine for smooth curve)
                    t = j / fade_out_samples
                    fade_val = 0.5 + 0.5 * np.cos(np.pi * t)
                    if mask[idx] > 0.5:  # Only apply to speech regions
                        mask[idx] = min(mask[idx], fade_val)
    
    # Light smoothing for overall coherence
    smoothed = np.zeros_like(mask)
    smoothed[0] = mask[0]
    for i in range(1, len(mask)):
        smoothed[i] = smoothing * smoothed[i-1] + (1 - smoothing) * mask[i]
    
    # Backward smooth for symmetric
    for i in range(len(mask) - 2, -1, -1):
        smoothed[i] = max(smoothed[i], smoothing * smoothed[i+1])
    
    return smoothed


# ============================================
# MAIN PROCESSOR CLASS
# ============================================
class AudioProcessorV12Blended:
    """
    V12 Blended Output Processor
    
    Blends CRNN (for speech) with V7 (for silence).
    """
    
    def __init__(self, model_path: str = None):
        import os
        _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Use absolute path for model
        if model_path is None:
            model_path = os.path.join(_BASE_DIR, 'core', 'crnn_attention.pth')
        
        self.sample_rate = 16000
        self.frame_size = 512
        self.hop_size = 256
        self.n_fft = 512
        self.n_bins = self.n_fft // 2 + 1
        
        # Import V7 processor
        from audio_processor_crnn_attention_latent import AudioProcessorHybrid16kNative
        self.v7_processor = AudioProcessorHybrid16kNative(model_path=model_path)
        
        # Import CRNN model
        from core.crnn_attention import CRNN_GRU_Attention
        import torch
        
        self.device = 'cpu'
        self.model = CRNN_GRU_Attention(
            input_size=5,
            gru_hidden=64,
            gru_layers=2
        ).to(self.device)
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.window = np.hanning(self.frame_size)
        
        print("[OK] V12 Blended Output Processor Initialized")
        print("     - Speech: CRNN output (better voice)")
        print("     - Silence: V7 output (digital black)")
    
    def stft(self, audio: np.ndarray) -> np.ndarray:
        n_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        spectrogram = np.zeros((n_frames, self.n_bins), dtype=np.complex128)
        
        for i in range(n_frames):
            start = i * self.hop_size
            frame = audio[start:start + self.frame_size] * self.window
            spectrogram[i] = np.fft.rfft(frame, n=self.n_fft)
        
        return spectrogram
    
    def istft(self, spectrogram: np.ndarray) -> np.ndarray:
        n_frames = spectrogram.shape[0]
        output_len = (n_frames - 1) * self.hop_size + self.frame_size
        
        output = np.zeros(output_len)
        window_sum = np.zeros(output_len)
        
        for i in range(n_frames):
            start = i * self.hop_size
            frame = np.real(np.fft.irfft(spectrogram[i], n=self.n_fft))[:self.frame_size]
            output[start:start + self.frame_size] += frame * self.window
            window_sum[start:start + self.frame_size] += self.window ** 2
        
        window_sum = np.maximum(window_sum, 1e-8)
        output /= window_sum
        
        return output
    
    def extract_features(self, spectrogram: np.ndarray) -> np.ndarray:
        n_frames, n_bins = spectrogram.shape
        magnitude = np.abs(spectrogram)
        power = magnitude ** 2
        
        features = np.zeros((n_frames, 5), dtype=np.float32)
        
        for i in range(n_frames):
            frame_power = power[i]
            frame_mag = magnitude[i]
            
            energy = np.sum(frame_power) + 1e-10
            features[i, 0] = np.log10(energy)
            
            freqs = np.arange(n_bins)
            centroid = np.sum(freqs * frame_mag) / (np.sum(frame_mag) + 1e-10)
            features[i, 1] = centroid / n_bins
            
            geo_mean = np.exp(np.mean(np.log(frame_mag + 1e-10)))
            arith_mean = np.mean(frame_mag) + 1e-10
            features[i, 2] = geo_mean / arith_mean
            
            cumsum = np.cumsum(frame_power)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
            features[i, 3] = rolloff_idx / n_bins
            
            if i > 0:
                features[i, 4] = np.mean(np.abs(magnitude[i] - magnitude[i-1]))
        
        return features
    
    def predict_spp_chunked(self, features: np.ndarray) -> tuple:
        import torch
        
        n_frames = len(features)
        chunk_size = 1875
        
        alpha_all = []
        spp_all = []
        
        for start in range(0, n_frames, chunk_size):
            end = min(start + chunk_size, n_frames)
            chunk_features = features[start:end]
            
            x = torch.FloatTensor(chunk_features).unsqueeze(0)
            
            with torch.no_grad():
                out, _, _ = self.model(x, None)
            
            out = out.squeeze(0).numpy()
            alpha_all.append(out[:, 0])
            spp_all.append(out[:, 1])
        
        alpha = np.concatenate(alpha_all)
        spp = np.concatenate(spp_all)
        
        return np.clip(alpha, 0.92, 0.995), np.clip(spp, 0, 1)
    
    def apply_wiener(self, spectrogram: np.ndarray, alpha_seq: np.ndarray) -> np.ndarray:
        """Apply standard Wiener filter."""
        n_frames = len(spectrogram)
        power_spectrum = np.abs(spectrogram) ** 2
        
        noise_est = np.percentile(power_spectrum, 10, axis=0)
        
        enhanced = np.zeros_like(spectrogram)
        prev_clean_sq = power_spectrum[0].copy()
        
        oversub = 2.0
        floor = 0.02
        
        for i in range(n_frames):
            noisy_power = power_spectrum[i]
            frame_alpha = alpha_seq[i] if i < len(alpha_seq) else 0.95
            
            noise_adj = noise_est * oversub
            gamma_snr = noisy_power / np.maximum(noise_adj, 1e-10)
            
            xi_prev = prev_clean_sq / np.maximum(noise_est, 1e-10)
            xi_ml = np.maximum(gamma_snr - 1, 0)
            xi = frame_alpha * xi_prev + (1 - frame_alpha) * xi_ml
            xi = np.maximum(xi, 1e-3)
            
            gain = xi / (1.0 + xi)
            gain = np.clip(gain, floor, 1.0)
            
            enhanced[i] = gain * spectrogram[i]
            prev_clean_sq = np.abs(enhanced[i]) ** 2
        
        return enhanced
    
    def process(self, audio: np.ndarray, sr: int = 16000):
        """Blended processing."""
        start_time = time.perf_counter()
        original_length = len(audio)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val < 1e-6:
            max_val = 1.0
        audio_norm = audio / max_val
        
        # ============================================
        # RUN V7 PROCESSOR
        # ============================================
        print("  [V7] Running V7 Processor...")
        v7_output, v7_info = self.v7_processor.process(audio_norm, sr)
        print(f"       > V7 Speed: {v7_info.get('speed', 0):.1f}x")
        
        # ============================================
        # RUN CRNN ENHANCEMENT
        # ============================================
        print("  [CRNN] Running CRNN Enhancement...")
        spectrogram = self.stft(audio_norm)
        features = self.extract_features(spectrogram)
        alpha_seq, _ = self.predict_spp_chunked(features)
        
        enhanced_spectrogram = self.apply_wiener(spectrogram, alpha_seq)
        crnn_output = self.istft(enhanced_spectrogram)
        
        # Match lengths
        min_len = min(len(v7_output), len(crnn_output))
        v7_output = v7_output[:min_len]
        crnn_output = crnn_output[:min_len]
        
        # ============================================
        # CREATE BLEND MASK FROM V7 OUTPUT
        # ============================================
        print("  [BLEND] Creating blend mask...")
        blend_mask = create_blend_mask(v7_output, sr)
        blend_mask = blend_mask[:min_len]
        
        speech_pct = np.mean(blend_mask) * 100
        print(f"          > Speech (CRNN): {speech_pct:.1f}%")
        print(f"          > Silence (V7): {100-speech_pct:.1f}%")
        
        # ============================================
        # BLEND OUTPUTS
        # ============================================
        print("  [OUTPUT] Blending outputs...")
        # Where mask=1: use CRNN (speech)
        # Where mask=0: use V7 (silence/digital black)
        blended = blend_mask * crnn_output + (1 - blend_mask) * v7_output
        
        # Scale back
        blended = blended * max_val
        
        # Soft limiter
        threshold = 0.95
        blended = np.where(
            np.abs(blended) > threshold,
            np.sign(blended) * (threshold + np.tanh((np.abs(blended) - threshold) * 2) * (1 - threshold)),
            blended
        )
        
        # Match original length
        if len(blended) > original_length:
            blended = blended[:original_length]
        elif len(blended) < original_length:
            blended = np.pad(blended, (0, original_length - len(blended)))
        
        total_time = time.perf_counter() - start_time
        audio_duration = original_length / sr
        speed = audio_duration / total_time
        
        info = {
            'speed': speed,
            'speech_ratio': speech_pct / 100,
        }
        
        return blended, info


# ============================================
# MAIN
# ============================================
def main():
    import argparse
    import soundfile as sf
    
    parser = argparse.ArgumentParser(description='V12 Blended Output Enhancement')
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('output', nargs='?', help='Output audio file')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    print("=" * 60)
    print("V12 Blended Output Enhancement")
    print("Speech = CRNN | Silence = V7 (Digital Black)")
    print("=" * 60)
    
    print(f"\n[Loading] {args.input}")
    audio, sr = sf.read(args.input)
    
    if sr != 16000:
        print(f"  Resampling {sr}Hz -> 16000Hz...")
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
    
    is_stereo = audio.ndim == 2
    
    if is_stereo:
        left = audio[:, 0]
        right = audio[:, 1]
        print(f"  Duration: {len(left)/sr:.1f}s, Stereo")
    else:
        left = audio
        right = None
        print(f"  Duration: {len(left)/sr:.1f}s, Mono")
    
    processor = AudioProcessorV12Blended()
    
    print("\n[Enhancing...]")
    
    if is_stereo:
        # ตรวจสอบว่ามีเสียงทั้ง 2 channel หรือไม่
        left_rms = np.sqrt(np.mean(left ** 2))
        right_rms = np.sqrt(np.mean(right ** 2))
        
        # ถ้า channel หนึ่งเงียบ (RMS < 5% ของอีก channel)
        if left_rms > 0 and right_rms / left_rms < 0.05:
            print("\n  [Stereo] Detected: Right channel is silent, using Left only")
            mono_mix = left
        elif right_rms > 0 and left_rms / right_rms < 0.05:
            print("\n  [Stereo] Detected: Left channel is silent, using Right only")
            mono_mix = right
        else:
            print("\n  [Stereo→Mono] Mixing L+R for consistent gating...")
            mono_mix = (left + right) / 2
        
        # Process as mono
        print("\n  === Processing Mono ===")
        enhanced_mono, info = processor.process(mono_mix, sr)
        
        # Output as dual-mono (L = R)
        print("  [Mono→Stereo] Creating dual-mono output...")
        enhanced = np.column_stack([enhanced_mono, enhanced_mono])
    else:
        enhanced, info = processor.process(left, sr)
    
    if args.output:
        output_path = args.output
    else:
        output_path = input_path.stem + "_v12" + ".wav"
    
    sf.write(output_path, enhanced.astype(np.float32), sr)
    
    print("\n" + "=" * 60)
    print("[Complete]")
    print(f"  Output: {output_path}")
    print(f"  Speed: {info['speed']:.1f}x realtime")
    print(f"  Speech: {info['speech_ratio']:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
