#!/usr/bin/env python
"""
Audio Enhancement CLI - V12/V14 Edition
========================================
Simplified entry point for forensic audio enhancement.

Usage:
    python enhance_v12.py <input.mp3> [output.wav]
    python enhance_v12.py <input.mp3> --auto          # Auto: V12 or 2-pass
    python enhance_v12.py <input.mp3> --hf            # V14: + HF Reduction
    python enhance_v12.py <input.mp3> --2pass         # 2-pass: V7 → V12

Modes:
    - Default (V12): Digital Black + CRNN Blend + Fade In/Out
    - HF Mode (V14): V12 + High Frequency Reduction (70%)
    - 2-Pass Mode:   V7 (pre-clean) → V12 (final) for high noise
    - Auto Mode:     Analyze source and choose V12 or 2-pass
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# ===========================================
# AUTO-MODE: Source Analysis Functions
# ===========================================

def noise_floor_db(audio, sr=16000, percentile=5):
    """Measure noise floor level in dB."""
    frame_size = int(0.03 * sr)  # 30ms
    hop_size = frame_size // 2
    
    n_frames = (len(audio) - frame_size) // hop_size
    if n_frames <= 0:
        return -60.0
        
    rms_values = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]
        rms_values[i] = np.sqrt(np.mean(frame ** 2))
    
    noise_floor = np.percentile(rms_values, percentile)
    
    if noise_floor < 1e-10:
        return -80.0
    
    return 20 * np.log10(noise_floor)


def silence_ratio(audio, sr=16000, threshold_db=-40):
    """Calculate percentage of audio below threshold."""
    frame_size = int(0.03 * sr)
    hop_size = frame_size // 2
    
    n_frames = (len(audio) - frame_size) // hop_size
    if n_frames <= 0:
        return 0.0
        
    threshold_linear = 10 ** (threshold_db / 20)
    silent_count = 0
    
    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < threshold_linear:
            silent_count += 1
    
    return silent_count / n_frames


def analyze_auto_mode(audio, sr=16000):
    """Analyze source and select best mode.
    
    Returns: (mode, needs_hf, analysis_info)
        mode: 'v14' or '2pass'
        needs_hf: True if --hf should be added
    """
    nf = noise_floor_db(audio, sr)
    sr_pct = silence_ratio(audio, sr) * 100
    
    analysis = {
        'noise_floor_db': nf,
        'silence_pct': sr_pct,
    }
    
    # Thresholds based on calibration:
    # NoiseFloor > -30 → 2-pass + HF
    # NoiseFloor > -35 → 2-pass
    # Otherwise → V14 (default)
    
    if nf > -30:
        mode = '2pass'
        needs_hf = True
        reason = f"Very noisy ({nf:.1f} dB) -> 2-pass + HF"
    elif nf > -35:
        mode = '2pass'
        needs_hf = False
        reason = f"Noisy ({nf:.1f} dB) -> 2-pass"
    else:
        mode = 'v14'
        needs_hf = True  # V14 always has HF
        reason = f"Clean ({nf:.1f} dB) -> V14"
    
    analysis['mode'] = mode
    analysis['needs_hf'] = needs_hf
    analysis['reason'] = reason
    
    return mode, needs_hf, analysis


def main():
    parser = argparse.ArgumentParser(
        description='Audio Enhancement V12/V14/V18',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python enhance_v12.py recording.mp3                # V12 (default)
    python enhance_v12.py recording.mp3 --auto         # Auto: V12 or 2-pass
    python enhance_v12.py recording.mp3 --hf           # V14 (+ HF reduction)
    python enhance_v12.py recording.mp3 --2pass        # 2-pass (V7 → V12)
    python enhance_v12.py recording.mp3 --2pass --hf   # 2-pass + V14
    python enhance_v12.py recording.mp3 --fast         # V18 Fast (smart auto)
        """
    )
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('output', nargs='?', help='Output audio file')
    parser.add_argument('--auto', action='store_true', help='Auto-select V12 or 2-pass')
    parser.add_argument('--hf', action='store_true', help='Enable HF Reduction (V14)')
    parser.add_argument('--2pass', dest='two_pass', action='store_true', help='2-Pass mode')
    parser.add_argument('--fast', action='store_true', help='Fast mode (V18 Smart)')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)
    
    # Load audio
    print("=" * 60)
    print(f"[Loading] {args.input}")
    audio, sr = sf.read(args.input)
    
    # Resample to 16kHz
    if sr != 16000:
        print(f"  Resampling {sr}Hz -> 16000Hz...")
        import math
        gcd = math.gcd(16000, sr)
        up, down = 16000 // gcd, sr // gcd
        
        if audio.ndim == 2:
            audio = np.column_stack([resample_poly(audio[:, 0], up, down),
                                     resample_poly(audio[:, 1], up, down)])
        else:
            audio = resample_poly(audio, up, down)
        sr = 16000
    
    is_stereo = audio.ndim == 2
    duration = len(audio) if audio.ndim == 1 else len(audio[:, 0])
    print(f"  Duration: {duration/sr:.1f}s, {'Stereo' if is_stereo else 'Mono'}")
    
    # ===========================================
    # AUTO-MODE: Analyze and decide
    # ===========================================
    if args.auto:
        print("\n[AUTO] Analyzing source...")
        mono = (audio[:, 0] + audio[:, 1]) / 2 if is_stereo else audio
        mode, needs_hf, analysis = analyze_auto_mode(mono, sr)
        
        print(f"  Noise Floor: {analysis['noise_floor_db']:.1f} dB")
        print(f"  Silence: {analysis['silence_pct']:.1f}%")
        print(f"  -> Selected: {analysis['reason']}")
        
        if mode == '2pass':
            args.two_pass = True
        if needs_hf:
            args.hf = True
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.fast:
            suffix = "_v18"
        elif args.two_pass and args.hf:
            suffix = "_2pass_hf"
        elif args.two_pass:
            suffix = "_2pass"
        elif args.hf:
            suffix = "_v14"
        else:
            suffix = "_v12"
        output_path = input_path.stem + suffix + ".wav"
    
    # Print mode
    print("\n" + "=" * 60)
    if args.fast:
        print("V18 Fast Enhancement (Smart Auto)")
        print("  = Auto V7 passes + Blend + HF")
    elif args.two_pass:
        print("2-Pass Enhancement (High Noise)")
        print("  Pass 1: V7 (CRNN Latent) - Pre-clean")
        print("  Pass 2: V12 - Final (Digital Black)")
        if args.hf:
            print("  + HF Reduction (V14)")
    elif args.hf:
        print("V14 Enhancement (V12 + HF Reduction)")
    else:
        print("V12 Enhancement (Digital Black + CRNN Blend)")
    print("=" * 60)
    
    # ============================================
    # PROCESSING
    # ============================================
    print("\n[Enhancing...]")
    
    if args.fast:
        # V18 Fast Mode
        from audio_processor_v18_smart import AudioProcessorV18Smart
        processor = AudioProcessorV18Smart()
        
        if is_stereo:
            print("\n  [Stereo→Mono] Mixing L+R...")
            mono_mix = (audio[:, 0] + audio[:, 1]) / 2
            enhanced_mono, info = processor.process(mono_mix, sr)
            print("  [Mono→Stereo] Creating dual-mono output...")
            enhanced = np.column_stack([enhanced_mono, enhanced_mono])
        else:
            enhanced, info = processor.process(audio, sr)
    
    elif args.two_pass:
        print("\n" + "=" * 40)
        print("PASS 1: V7 (Pre-cleaning)")
        print("=" * 40)
        
        from audio_processor_crnn_attention_latent import AudioProcessorHybrid16kNative
        v7 = AudioProcessorHybrid16kNative()
        
        if is_stereo:
            mono_mix = (audio[:, 0] + audio[:, 1]) / 2
            print("\n  [Stereo→Mono] Mixing L+R...")
            pre_cleaned, _ = v7.process(mono_mix, sr)
        else:
            pre_cleaned, _ = v7.process(audio, sr)
        
        print("\n" + "=" * 40)
        print("PASS 2: V12/V14 (Final)")
        print("=" * 40)
        
        if args.hf:
            from audio_processor_v14_combined import AudioProcessorV14Combined
            processor = AudioProcessorV14Combined()
        else:
            from audio_processor_v12_blended import AudioProcessorV12Blended
            processor = AudioProcessorV12Blended()
        
        enhanced_mono, info = processor.process(pre_cleaned, sr)
        
        if is_stereo:
            print("  [Mono→Stereo] Creating dual-mono output...")
            enhanced = np.column_stack([enhanced_mono, enhanced_mono])
        else:
            enhanced = enhanced_mono
    else:
        if args.hf:
            from audio_processor_v14_combined import AudioProcessorV14Combined
            processor = AudioProcessorV14Combined()
        else:
            from audio_processor_v12_blended import AudioProcessorV12Blended
            processor = AudioProcessorV12Blended()
        
        if is_stereo:
            print("\n  [Stereo→Mono] Mixing L+R for consistent gating...")
            mono_mix = (audio[:, 0] + audio[:, 1]) / 2
            print("\n  === Processing Mono ===")
            enhanced_mono, info = processor.process(mono_mix, sr)
            print("  [Mono→Stereo] Creating dual-mono output...")
            enhanced = np.column_stack([enhanced_mono, enhanced_mono])
        else:
            enhanced, info = processor.process(audio, sr)
    
    # Save
    sf.write(output_path, enhanced.astype(np.float32), sr)
    
    print("\n" + "=" * 60)
    print("[Complete]")
    print(f"  Output: {output_path}")
    print(f"  Speed: {info['speed']:.1f}x realtime")
    if 'speech_ratio' in info:
        print(f"  Speech: {info['speech_ratio']:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
