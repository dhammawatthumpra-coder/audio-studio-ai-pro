"""
Audio Processor V19 - Digital Black Cleanup
============================================
Post-processor สำหรับเก็บกวาด residual noise จากไฟล์ที่ผ่าน V18 แล้ว

Design:
- Input: Enhanced audio (ผ่าน V18 แล้ว)
- Output: Same audio แต่ silence regions เป็น digital black

Algorithm:
1. Analyze → measure noise floor & speech level
2. Detect speech regions (energy + ZCR based)
3. Extend margins (attack/release) to protect speech tails
4. Apply digital black with smooth crossfade
"""

import numpy as np
import time
from pathlib import Path
from numba import jit


# ============================================
# NUMBA-ACCELERATED FUNCTIONS
# ============================================
@jit(nopython=True, cache=True)
def compute_frame_energy(audio, frame_size, hop_size):
    """Compute energy per frame (RMS)."""
    n_frames = (len(audio) - frame_size) // hop_size + 1
    energy = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]
        energy[i] = np.sqrt(np.mean(frame ** 2))
    
    return energy


@jit(nopython=True, cache=True)
def compute_zcr(audio, frame_size, hop_size):
    """Compute Zero-Crossing Rate per frame."""
    n_frames = (len(audio) - frame_size) // hop_size + 1
    zcr = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]
        
        crossings = 0
        for j in range(1, len(frame)):
            if (frame[j-1] >= 0 and frame[j] < 0) or (frame[j-1] < 0 and frame[j] >= 0):
                crossings += 1
        
        zcr[i] = crossings / frame_size
    
    return zcr


@jit(nopython=True, cache=True)
def extend_mask(mask, attack_frames, release_frames):
    """Extend speech regions with attack (before) and release (after) margins."""
    extended = mask.copy()
    n = len(mask)
    
    # Forward pass: extend speech backwards (attack)
    in_speech_region = False
    frames_since_speech = 0
    
    for i in range(n - 1, -1, -1):
        if mask[i] > 0.5:
            in_speech_region = True
            frames_since_speech = 0
        else:
            if in_speech_region:
                frames_since_speech += 1
                if frames_since_speech <= attack_frames:
                    extended[i] = 1.0
                else:
                    in_speech_region = False
    
    # Backward pass: extend speech forwards (release)
    in_speech_region = False
    frames_since_speech = 0
    
    for i in range(n):
        if mask[i] > 0.5:
            in_speech_region = True
            frames_since_speech = 0
        else:
            if in_speech_region:
                frames_since_speech += 1
                if frames_since_speech <= release_frames:
                    extended[i] = 1.0
                else:
                    in_speech_region = False
    
    return extended


@jit(nopython=True, cache=True)
def apply_crossfade_mask(mask, fade_frames):
    """Apply smooth crossfade at speech/silence boundaries."""
    n = len(mask)
    smooth = mask.copy()
    
    for i in range(n):
        if mask[i] > 0.5:
            # Check if near boundary (look back)
            for j in range(1, fade_frames + 1):
                if i - j >= 0 and mask[i - j] < 0.5:
                    # Fade in from silence
                    fade_pos = j / fade_frames
                    smooth[i] = min(smooth[i], fade_pos)
                    break
            
            # Check if near boundary (look forward)
            for j in range(1, fade_frames + 1):
                if i + j < n and mask[i + j] < 0.5:
                    # Fade out to silence
                    fade_pos = j / fade_frames
                    smooth[i] = min(smooth[i], fade_pos)
                    break
    
    return smooth


@jit(nopython=True, cache=True)
def filter_low_amp_short_bursts(mask, energy, min_duration_frames, max_amplitude_ratio):
    """
    Remove regions that are BOTH low amplitude AND short duration.
    
    Conditions to remove:
    1. Duration < min_duration_frames
    2. AND avg amplitude in region < max_amplitude_ratio * peak
    
    This preserves:
    - Consonants (high amplitude + short) 
    - Soft speech (low amplitude + long)
    
    Only removes:
    - Noise (low amplitude + short)
    """
    n = len(mask)
    filtered = mask.copy()
    
    # Get peak energy for reference
    peak_energy = 0.0
    for i in range(len(energy)):
        if energy[i] > peak_energy:
            peak_energy = energy[i]
    
    if peak_energy < 1e-10:
        return filtered
    
    amplitude_threshold = peak_energy * max_amplitude_ratio
    
    i = 0
    while i < n:
        if mask[i] > 0.5:
            # Found start of speech region
            start = i
            # Find end and calculate AVERAGE amplitude (total energy / duration)
            total_energy = 0.0
            frame_count = 0
            while i < n and mask[i] > 0.5:
                if i < len(energy):
                    total_energy += energy[i]
                    frame_count += 1
                i += 1
            end = i
            
            duration = end - start
            
            # Use AVERAGE amplitude (total energy / frames) instead of max
            avg_amp_in_region = total_energy / max(1, frame_count)
            
            # Check BOTH conditions: short AND low AVERAGE amplitude
            is_short = duration < min_duration_frames
            is_low_avg_amp = avg_amp_in_region < amplitude_threshold
            
            # Remove if BOTH: low average amp + short
            if is_short and is_low_avg_amp:
                for j in range(start, end):
                    filtered[j] = 0.0
        else:
            i += 1
    
    return filtered


@jit(nopython=True, cache=True)
def filter_isolated_in_silence(mask, energy, min_silence_frames, max_amp_ratio, max_duration_frames):
    """
    Remove regions that are ISOLATED in long silence periods.
    
    If a region has long silence (> min_silence_frames) on BOTH sides,
    and is shorter than max_duration_frames with low average amplitude,
    it's probably noise, not speech.
    
    Real speech rarely appears alone in the middle of long silence gaps.
    """
    n = len(mask)
    filtered = mask.copy()
    
    # Get peak energy
    peak_energy = 0.0
    for i in range(len(energy)):
        if energy[i] > peak_energy:
            peak_energy = energy[i]
    
    if peak_energy < 1e-10:
        return filtered
    
    amp_threshold = peak_energy * max_amp_ratio
    
    i = 0
    while i < n:
        if mask[i] > 0.5:
            start = i
            total_energy = 0.0
            frame_count = 0
            
            while i < n and mask[i] > 0.5:
                if i < len(energy):
                    total_energy += energy[i]
                    frame_count += 1
                i += 1
            end = i
            
            duration = end - start
            avg_amp = total_energy / max(1, frame_count)
            
            # Check if isolated in long silence
            # Count silence frames before
            silence_before = 0
            j = start - 1
            while j >= 0 and mask[j] < 0.5:
                silence_before += 1
                j -= 1
            
            # Count silence frames after
            silence_after = 0
            j = end
            while j < n and mask[j] < 0.5:
                silence_after += 1
                j += 1
            
            # If surrounded by long silence on BOTH sides
            is_isolated = silence_before >= min_silence_frames and silence_after >= min_silence_frames
            is_short_enough = duration < max_duration_frames
            is_low_amp = avg_amp < amp_threshold
            
            # Remove if isolated in silence AND short AND low amp
            if is_isolated and is_short_enough and is_low_amp:
                for k in range(start, end):
                    filtered[k] = 0.0
        else:
            i += 1
    
    return filtered


# ============================================
# V19 CLEANUP PROCESSOR
# ============================================
class AudioProcessorV19Cleanup:
    """
    V19 Digital Black Cleanup Processor
    
    Post-processes V18 output to achieve true digital black in silence regions.
    """
    
    def __init__(
        self,
        frame_ms: float = 10.0,
        attack_margin_ms: float = 40.0,
        release_margin_ms: float = 120.0,
        threshold_offset_db: float = -6.0,
        use_zcr: bool = True,
        crossfade_ms: float = 5.0,
        mode: str = "ultra",  # soft, normal, aggressive, ultra (default: ultra)
        # Absolute threshold for enhanced audio (relative to peak)
        speech_threshold_ratio: float = 0.02,  # 2% of peak = speech
        # Minimum duration filter (remove short spikes)
        min_duration_ms: float = 50.0,  # Speech shorter than this = spike noise
        # Max amplitude for noise (relative to peak)
        max_amplitude_ratio: float = 0.08  # 8% of peak = low amplitude noise
    ):
        self.sample_rate = 16000
        self.frame_ms = frame_ms
        self.attack_margin_ms = attack_margin_ms
        self.release_margin_ms = release_margin_ms
        self.threshold_offset_db = threshold_offset_db
        self.use_zcr = use_zcr
        self.crossfade_ms = crossfade_ms
        self.mode = mode
        self.speech_threshold_ratio = speech_threshold_ratio
        self.min_duration_ms = min_duration_ms
        self.max_amplitude_ratio = max_amplitude_ratio
        
        # Mode-specific adjustments
        if mode == "soft":
            self.threshold_offset_db = -9.0  # More permissive
            self.attack_margin_ms = 60.0
            self.release_margin_ms = 180.0
            self.speech_threshold_ratio = 0.01  # 1% of peak
            self.min_duration_ms = 30.0  # Allow shorter speech
        elif mode == "aggressive":
            self.threshold_offset_db = -3.0  # Stricter
            self.attack_margin_ms = 25.0
            self.release_margin_ms = 80.0
            self.speech_threshold_ratio = 0.05  # 5% of peak
            self.min_duration_ms = 60.0
        elif mode == "ultra":
            # Ultra: Conservative speech, aggressive noise filter
            self.threshold_offset_db = -6.0
            self.attack_margin_ms = 50.0
            self.release_margin_ms = 120.0
            self.speech_threshold_ratio = 0.02  # 2% - keep all speech
            self.min_duration_ms = 400.0  # Filter bursts up to 400ms
            self.max_amplitude_ratio = 0.20  # 20% - catch more noise
        
        print(f"[OK] V19 Digital Black Cleanup Initialized")
        print(f"     Mode: {mode.upper()}")
        print(f"     Attack: {self.attack_margin_ms:.0f}ms, Release: {self.release_margin_ms:.0f}ms")
        print(f"     Speech threshold: {self.speech_threshold_ratio*100:.1f}% of peak")
        print(f"     Min speech duration: {self.min_duration_ms:.0f}ms")
    
    def analyze_input(self, audio: np.ndarray, sr: int) -> dict:
        """Analyze input to determine thresholds."""
        frame_size = int(sr * self.frame_ms / 1000)
        hop_size = frame_size // 2
        
        energy = compute_frame_energy(audio, frame_size, hop_size)
        
        # Use percentiles for robust estimation
        noise_floor = np.percentile(energy, 5)
        speech_level = np.percentile(energy, 90)
        peak_level = np.max(energy)
        
        # CRITICAL FIX: For enhanced audio, noise floor may be ~0
        # Use absolute threshold based on peak level instead
        absolute_threshold = peak_level * self.speech_threshold_ratio
        
        # Dynamic threshold between noise and speech
        if noise_floor > 1e-6 and speech_level > noise_floor * 2:
            # Good separation - use geometric mean
            relative_threshold = np.sqrt(noise_floor * speech_level)
            # Apply offset
            relative_threshold *= 10 ** (self.threshold_offset_db / 20)
        else:
            # Enhanced audio: noise floor is ~0, use absolute threshold
            relative_threshold = 0.0
        
        # Use the HIGHER of relative and absolute thresholds
        # This ensures we catch residual noise in enhanced audio
        threshold = max(relative_threshold, absolute_threshold)
        
        # ZCR stats for speech detection
        if self.use_zcr:
            zcr = compute_zcr(audio, frame_size, hop_size)
            zcr_threshold = np.percentile(zcr, 30)  # Speech has lower ZCR
        else:
            zcr_threshold = 0.0
        
        return {
            'noise_floor': noise_floor,
            'speech_level': speech_level,
            'peak_level': peak_level,
            'threshold': threshold,
            'absolute_threshold': absolute_threshold,
            'zcr_threshold': zcr_threshold,
            'snr_db': 20 * np.log10(speech_level / (noise_floor + 1e-10)),
            'using_absolute': relative_threshold < absolute_threshold
        }
    
    def detect_speech_regions(
        self, 
        audio: np.ndarray, 
        sr: int, 
        analysis: dict
    ) -> np.ndarray:
        """Detect speech regions and create binary mask."""
        frame_size = int(sr * self.frame_ms / 1000)
        hop_size = frame_size // 2
        
        energy = compute_frame_energy(audio, frame_size, hop_size)
        threshold = analysis['threshold']
        
        # Energy-based detection
        mask = (energy > threshold).astype(np.float64)
        
        # ZCR-based refinement (speech has characteristic ZCR range)
        if self.use_zcr:
            zcr = compute_zcr(audio, frame_size, hop_size)
            # Speech typically has lower ZCR than noise
            # But not too low (silence has very low)
            zcr_low = 0.01
            zcr_high = analysis['zcr_threshold'] * 3
            
            # Combine: high energy AND reasonable ZCR = speech
            zcr_valid = (zcr > zcr_low) & (zcr < zcr_high)
            
            # Only apply ZCR constraint where energy is borderline
            borderline = (energy > threshold * 0.5) & (energy < threshold * 2)
            mask[borderline & ~zcr_valid] = 0.0
        
        return mask
    
    def process(self, audio: np.ndarray, sr: int = 16000) -> tuple:
        """
        Main processing function.
        
        Args:
            audio: Input audio (should be V18 output)
            sr: Sample rate
            
        Returns:
            Tuple[np.ndarray, dict]: (cleaned_audio, info)
        """
        start_time = time.perf_counter()
        original_length = len(audio)
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val < 1e-6:
            return audio, {'skipped': True, 'reason': 'silent input'}
        audio_norm = audio / max_val
        
        # Step 1: Analyze input
        print("  [1/4] Analyzing input...")
        analysis = self.analyze_input(audio_norm, sr)
        print(f"        Peak: {analysis['peak_level']:.4f}, Speech: {analysis['speech_level']:.4f}")
        if analysis['using_absolute']:
            print(f"        [ENHANCED MODE] Using absolute threshold: {analysis['threshold']:.6f}")
        else:
            print(f"        Noise: {analysis['noise_floor']:.6f}, Threshold: {analysis['threshold']:.6f}")
        
        # Step 2: Detect speech regions
        print("  [2/4] Detecting speech regions...")
        frame_size = int(sr * self.frame_ms / 1000)
        hop_size = frame_size // 2
        
        mask = self.detect_speech_regions(audio_norm, sr, analysis)
        
        # Step 3: Filter low-amplitude short bursts (noise spikes)
        print("  [3/5] Filtering low-amp short bursts...")
        min_duration_frames = int(self.min_duration_ms / self.frame_ms)
        
        # Get energy for amplitude checking
        frame_size = int(sr * self.frame_ms / 1000)
        hop_size = frame_size // 2
        energy = compute_frame_energy(audio_norm, frame_size, hop_size)
        
        original_mask = mask.copy()
        mask = filter_low_amp_short_bursts(mask, energy, min_duration_frames, self.max_amplitude_ratio)
        
        # Count removed spikes
        spikes_removed = int(np.sum(original_mask > 0.5) - np.sum(mask > 0.5))
        if spikes_removed > 0:
            print(f"        Removed {spikes_removed} noise frames (low amp + short)")
        
        # Step 3.5: Filter isolated regions in long silence gaps
        print("  [3.5/5] Filtering isolated-in-silence...")
        min_silence_frames = int(500.0 / self.frame_ms)  # 500ms silence threshold
        max_isolated_duration = int(800.0 / self.frame_ms)  # Up to 800ms
        isolated_amp_ratio = 0.25  # 25% of peak
        
        mask_before = mask.copy()
        mask = filter_isolated_in_silence(mask, energy, min_silence_frames, isolated_amp_ratio, max_isolated_duration)
        
        isolated_removed = int(np.sum(mask_before > 0.5) - np.sum(mask > 0.5))
        if isolated_removed > 0:
            print(f"        Removed {isolated_removed} isolated-in-silence frames")
        
        # Step 4: Extend margins
        print("  [4/5] Extending margins...")
        attack_frames = int(self.attack_margin_ms / self.frame_ms)
        release_frames = int(self.release_margin_ms / self.frame_ms)
        
        mask = extend_mask(mask, attack_frames, release_frames)
        
        # Apply crossfade for smooth transitions
        fade_frames = max(1, int(self.crossfade_ms / self.frame_ms))
        mask = apply_crossfade_mask(mask, fade_frames)
        
        speech_pct = np.mean(mask) * 100
        print(f"        Speech: {speech_pct:.1f}%")
        
        # Step 4: Apply digital black
        print("  [4/4] Applying digital black...")
        
        # Expand mask to sample level
        n_frames = len(mask)
        sample_mask = np.zeros(original_length)
        
        for i in range(n_frames):
            start = i * hop_size
            end = min(start + frame_size, original_length)
            # Use max to handle overlap correctly
            sample_mask[start:end] = np.maximum(sample_mask[start:end], mask[i])
        
        # Apply mask
        cleaned = audio * sample_mask
        
        # Count digital black samples
        black_samples = np.sum(sample_mask < 0.01)
        black_pct = black_samples / original_length * 100
        print(f"        Digital Black: {black_pct:.1f}%")
        
        elapsed = time.perf_counter() - start_time
        
        info = {
            'speed': (original_length / sr) / elapsed,
            'speech_pct': speech_pct,
            'black_pct': black_pct,
            'analysis': analysis,
            'mode': self.mode
        }
        
        return cleaned, info


# ============================================
# MAIN ENTRY POINT
# ============================================
def main():
    import argparse
    import soundfile as sf
    from scipy.signal import resample_poly
    import math
    
    parser = argparse.ArgumentParser(description='V19 Digital Black Cleanup')
    parser.add_argument('input', help='Input audio file (V18 output)')
    parser.add_argument('output', nargs='?', help='Output audio file')
    parser.add_argument('--mode', choices=['soft', 'normal', 'aggressive', 'ultra'], 
                        default='ultra', help='Cleanup mode (default: ultra)')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    print("=" * 60)
    print("V19 Digital Black Cleanup")
    print("=" * 60)
    
    # Load audio
    print(f"\n[Loading] {args.input}")
    audio, sr = sf.read(args.input)
    
    # Resample to 16k if needed
    if sr != 16000:
        print(f"  Resampling {sr}Hz -> 16000Hz...")
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
    is_stereo = audio.ndim == 2
    
    if is_stereo:
        left = audio[:, 0]
        right = audio[:, 1]
        print(f"  Duration: {len(left)/sr:.1f}s, Stereo")
    else:
        left = audio
        right = None
        print(f"  Duration: {len(left)/sr:.1f}s, Mono")
    
    # Process
    processor = AudioProcessorV19Cleanup(mode=args.mode)
    
    print("\n[Processing...]")
    
    if is_stereo:
        print("\n  === Left Channel ===")
        cleaned_l, info_l = processor.process(left, sr)
        print("\n  === Right Channel ===")
        cleaned_r, info_r = processor.process(right, sr)
        cleaned = np.column_stack([cleaned_l, cleaned_r])
        info = info_l
    else:
        cleaned, info = processor.process(left, sr)
    
    # Output
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.stem) + "_v19_cleanup" + input_path.suffix
    
    sf.write(output_path, cleaned.astype(np.float32), sr)
    
    print("\n" + "=" * 60)
    print("[Complete]")
    print(f"  Output: {output_path}")
    print(f"  Speed: {info['speed']:.1f}x realtime")
    print(f"  Speech: {info['speech_pct']:.1f}%")
    print(f"  Digital Black: {info['black_pct']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
