"""
Audio Processor V14 - V12 Base + V13 HF Reduction
=================================================
รวมจุดแข็งของ V12 และ V13:

Pipeline:
1. Run V12 (V7 + CRNN Blend + Fade In/Out) → v12_output
2. สำหรับช่วง Speech เท่านั้น:
   - Apply V13's Spectral Fusion เพื่อลด HF Hiss
3. ช่วง Silence: คง Digital Black จาก V12

ผลลัพธ์:
- Silence: Digital Black (จาก V12)
- Speech: Natural voice + Reduced HF hiss (จาก V13 technique)
"""

import numpy as np
import time
from pathlib import Path
from numba import jit
from scipy.signal import resample_poly
import torch


# ============================================
# NUMBA: HF REDUCTION IN SPEECH REGIONS
# ============================================
@jit(nopython=True, cache=True, nogil=True)
def apply_hf_reduction_numba(
    spectrogram: np.ndarray,
    speech_mask: np.ndarray,
    n_bins: int,
    hf_start_bin: int = 120,
    hf_reduction: float = 0.6,    # ลด HF ลง 40% ในช่วง Speech
    hf_floor: float = 0.3         # ไม่ลดต่ำกว่านี้
) -> np.ndarray:
    """
    ลด High Frequency hiss เฉพาะในช่วง Speech
    
    ช่วง Silence: ไม่แตะต้อง (เพราะ V12 จัดการ Digital Black แล้ว)
    ช่วง Speech: ลด HF bins เพื่อลดเสียงซ่า
    """
    n_frames = spectrogram.shape[0]
    enhanced = spectrogram.copy()
    
    for i in range(n_frames):
        # ดูว่าเป็นช่วง Speech หรือไม่
        is_speech = speech_mask[i] > 0.5 if i < len(speech_mask) else False
        
        if is_speech:
            for j in range(hf_start_bin, n_bins):
                # ลด HF แบบ gradual (ยิ่งสูงยิ่งลด)
                freq_ratio = (j - hf_start_bin) / max(1, (n_bins - hf_start_bin))
                reduction = hf_reduction - (freq_ratio * 0.2)  # ยิ่งสูงยิ่งลดน้อยลงนิดหน่อย
                reduction = max(hf_floor, min(1.0, reduction))
                enhanced[i, j] = spectrogram[i, j] * reduction
    
    return enhanced


# ============================================
# CREATE SPEECH MASK (from V12's blend mask concept)
# ============================================
def create_speech_mask(
    v7_output: np.ndarray,
    sr: int,
    hop_size: int = 256,
    threshold_ratio: float = 0.005
) -> np.ndarray:
    """Create frame-level speech mask from V7 output."""
    global_peak = np.max(np.abs(v7_output))
    if global_peak < 1e-6:
        return np.zeros(len(v7_output) // hop_size + 1)
    
    threshold = global_peak * threshold_ratio
    
    n_frames = len(v7_output) // hop_size + 1
    mask = np.zeros(n_frames, dtype=np.float32)
    
    for i in range(n_frames):
        start = i * hop_size
        end = min(start + hop_size, len(v7_output))
        if end > start:
            frame_peak = np.max(np.abs(v7_output[start:end]))
            mask[i] = 1.0 if frame_peak > threshold else 0.0
    
    return mask


# ============================================
# MAIN PROCESSOR CLASS
# ============================================
class AudioProcessorV14Combined:
    """
    V14 Combined Processor
    
    = V12 Base (Digital Black + Noise Reduction + Fade)
    + V13 HF Reduction (Natural Speech)
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
        self.n_bins = self.n_fft // 2 + 1  # 257
        self.hf_start_bin = 120  # ~3.7kHz
        
        # Import V12 processor
        from audio_processor_v12_blended import AudioProcessorV12Blended
        self.v12_processor = AudioProcessorV12Blended(model_path=model_path)
        
        self.window = np.hanning(self.frame_size)
        
        print("[OK] V14 Combined Processor Initialized")
        print("     - Base: V12 (Digital Black + Noise Reduction)")
        print("     - Add: V13 HF Reduction (Natural Speech)")
    
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
    
    def process(self, audio: np.ndarray, sr: int = 16000):
        """V14 Combined processing."""
        start_time = time.perf_counter()
        original_length = len(audio)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val < 1e-6:
            max_val = 1.0
        audio_norm = audio / max_val
        
        # ============================================
        # STEP 1: RUN V12 AS BASE
        # ============================================
        print("  [V12] Running V12 Base Processor...")
        v12_output, v12_info = self.v12_processor.process(audio_norm, sr)
        
        # ============================================
        # STEP 2: CREATE SPEECH MASK FROM V12 OUTPUT
        # ============================================
        print("  [MASK] Creating Speech Mask...")
        speech_mask = create_speech_mask(v12_output, sr, self.hop_size)
        
        speech_pct = np.mean(speech_mask) * 100
        print(f"         > Speech: {speech_pct:.1f}%")
        
        # ============================================
        # STEP 3: APPLY HF REDUCTION TO V12 OUTPUT
        # ============================================
        print("  [HF] Applying HF Reduction in Speech...")
        
        # Convert to spectrogram
        v12_spec = self.stft(v12_output)
        
        # Apply HF reduction only in speech regions
        enhanced_spec = apply_hf_reduction_numba(
            v12_spec.astype(np.complex128),
            speech_mask.astype(np.float32),
            n_bins=self.n_bins,
            hf_start_bin=self.hf_start_bin,
            hf_reduction=0.3,  # ลด 70% (เหลือ 30%)
            hf_floor=0.2       # ไม่ลดต่ำกว่า 20%
        )
        
        # ============================================
        # STEP 4: RECONSTRUCT
        # ============================================
        print("  [iSTFT] Reconstructing...")
        enhanced = self.istft(enhanced_spec) * max_val
        
        # ============================================
        # STEP 5: LOUDNESS COMPENSATION
        # ============================================
        # ชดเชย amplitude ที่หายไปจากการลด HF
        v12_rms = np.sqrt(np.mean(v12_output ** 2)) + 1e-10
        enhanced_rms = np.sqrt(np.mean(enhanced ** 2)) + 1e-10
        
        if enhanced_rms > 0:
            compensation = v12_rms / enhanced_rms
            # จำกัดไม่ให้ compensate เกินไป
            compensation = min(compensation, 1.5)
            enhanced = enhanced * compensation
            print(f"  [LOUDNESS] Compensation: {compensation:.2f}x")
        
        # Soft limiter
        threshold = 0.95
        enhanced = np.where(
            np.abs(enhanced) > threshold,
            np.sign(enhanced) * (threshold + np.tanh((np.abs(enhanced) - threshold) * 2) * (1 - threshold)),
            enhanced
        )
        
        # Match original length
        if len(enhanced) > original_length:
            enhanced = enhanced[:original_length]
        elif len(enhanced) < original_length:
            enhanced = np.pad(enhanced, (0, original_length - len(enhanced)))
        
        total_time = time.perf_counter() - start_time
        audio_duration = original_length / sr
        speed = audio_duration / total_time
        
        info = {
            'speed': speed,
            'speech_ratio': speech_pct / 100,
        }
        
        return enhanced, info


# ============================================
# MAIN
# ============================================
def main():
    import argparse
    import soundfile as sf
    
    parser = argparse.ArgumentParser(description='V14 Combined Enhancement')
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('output', nargs='?', help='Output audio file')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    print("=" * 60)
    print("V14 Combined Enhancement")
    print("V12 Base + V13 HF Reduction")
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
    
    processor = AudioProcessorV14Combined()
    
    print("\n[Enhancing...]")
    
    if is_stereo:
        # Mono Processing สำหรับ Gate Sync (เหมือน V12)
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
        output_path = input_path.stem + "_v14" + ".wav"
    
    sf.write(output_path, enhanced.astype(np.float32), sr)
    
    print("\n" + "=" * 60)
    print("[Complete]")
    print(f"  Output: {output_path}")
    print(f"  Speed: {info['speed']:.1f}x realtime")
    print(f"  Speech: {info['speech_ratio']:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
