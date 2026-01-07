"""
Audio Enhancer V2 - Core Audio Processor
=========================================
Core engine สำหรับการประมวลผลเสียง แยกออกจาก GUI
"""

import os
import sys
import subprocess
import importlib.util
import numpy as np
from typing import Tuple, Callable, Optional, Dict, Any
from dataclasses import asdict
from scipy import signal as scipy_signal
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import wiener

import noisereduce as nr
from pedalboard import (
    Pedalboard, NoiseGate, Compressor, HighpassFilter, LowpassFilter,
    PeakFilter, Limiter, LowShelfFilter, HighShelfFilter, Gain
)

from constants import (
    DEFAULT_SAMPLE_RATE, CHUNK_SIZE_SECONDS, FFMPEG_CODECS,
    ProcessingMode, ProcessingConfig, DenoiseEngine
)

# Check for Silero/PyTorch availability
SILERO_AVAILABLE = False
SILERO_MODEL = None
try:
    import torch
    import torchaudio
    SILERO_AVAILABLE = True
except ImportError:
    pass  # torch/torchaudio not installed


class AudioProcessor:
    """
    Core Audio Processing Engine
    ============================
    แยกการประมวลผลเสียงออกจาก GUI เพื่อให้สามารถ:
    - ทดสอบได้ง่าย
    - ใช้ซ้ำในที่อื่นได้
    - ดูแลรักษาง่าย
    """
    
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """ตรวจสอบว่ามี FFmpeg ติดตั้งหรือไม่"""
        try:
            creation_flags = 0x08000000 if os.name == 'nt' else 0
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                creationflags=creation_flags
            )
            self.ffmpeg_available = result.returncode == 0
        except FileNotFoundError:
            self.ffmpeg_available = False
        return self.ffmpeg_available
    
    def process_file(
        self,
        input_path: str,
        output_path: str,
        config: ProcessingConfig,
        log_func: Optional[Callable[[str], None]] = None,
        progress_func: Optional[Callable[[float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> Tuple[bool, str]:
        """
        ประมวลผลไฟล์เสียงหนึ่งไฟล์
        
        Args:
            input_path: path ของไฟล์ต้นฉบับ
            output_path: path สำหรับบันทึกผลลัพธ์
            config: การตั้งค่าการประมวลผล
            log_func: function สำหรับ log ข้อความ
            progress_func: function สำหรับอัพเดท progress (0.0-1.0)
            cancel_check: function ที่ return True เมื่อต้องการยกเลิก
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        log = log_func or (lambda x: None)
        progress = progress_func or (lambda x: None)
        is_cancelled = cancel_check or (lambda: False)
        
        try:
            # Step 1: Read file (10%)
            log(f"⏳ กำลังอ่าน: {os.path.basename(input_path)}...")
            progress(0.1)
            
            data, sr, is_true_stereo = self._read_audio(input_path)
            
            if is_true_stereo:
                duration = len(data) / sr
                log(f"   ✓ โหลดเสร็จ: {duration:.2f} วินาที (True Stereo)")
            else:
                duration = len(data) / sr
                log(f"   ✓ โหลดเสร็จ: {duration:.2f} วินาที (Mono)")
            progress(0.15)
            
            # Step 2: Fix Mono Channel (20%) - deprecated, handled in _read_audio
            progress(0.2)
            
            # Helper function to process single channel
            def process_channel(channel_data, channel_name=""):
                processed = channel_data.copy()
                
                # Step 1: Low Cut (ตัด rumble - basic pre-processing)
                if config.dynamics.lowcut_enabled:
                    if channel_name:
                        log(f"   [{channel_name}] Low Cut...")
                    processed = self._apply_lowcut(processed, sr, config.dynamics.lowcut_freq, log if not channel_name else lambda x: None)
                
                # Step 2: Transient Suppression (ลด spike ก่อนเข้า AI)
                transient_enabled = getattr(config.denoise, 'transient_suppression_enabled', True)
                if transient_enabled:
                    if channel_name:
                        log(f"⚡ [{channel_name}] Transient Suppression...")
                    processed = self._apply_transient_suppression(processed, sr, log if not channel_name else lambda x: None)
                
                # Step 3: Denoise Engine (AI ทำงานกับสัญญาณที่สะอาด)
                if config.denoise.enabled:
                    # Show denoise log only for first channel (L) or mono
                    show_denoise_log = (channel_name == "" or channel_name == "L")
                    processed = self._apply_denoise(processed, sr, config.denoise, log if show_denoise_log else lambda x: None)
                
                # Step 4: De-Hum (post-processing - ลบ residual hum)
                if config.eq.dehum_enabled:
                    if channel_name:
                        log(f"   [{channel_name}] De-Hum...")
                    processed = self._apply_dehum(processed, sr, log if not channel_name else lambda x: None)
                
                # Step 5: Anti-Drone (polish residual drone)
                if config.denoise.anti_drone_enabled:
                    strength = getattr(config.denoise, 'anti_drone_strength', 0.75)
                    processed = self._apply_anti_drone(processed, sr, config.denoise.anti_drone_threshold, strength, log if not channel_name else lambda x: None)
                
                return processed
            
            # Process stereo or mono
            if is_true_stereo:
                log("🎧 ประมวลผล Stereo (L/R แยก)...")
                
                # Process Left channel
                log("   [L] Processing Left...")
                left = process_channel(data[:, 0], "L")
                progress(0.35)
                
                # Check for cancel between channels
                if is_cancelled():
                    return False, "ยกเลิกโดยผู้ใช้"
                
                # Process Right channel
                log("   [R] Processing Right...")
                right = process_channel(data[:, 1], "R")
                progress(0.55)
                
                # Combine back to stereo
                data = np.column_stack([left, right])
                log("   ✓ Stereo processing completed")
            else:
                # Mono processing
                data = process_channel(data)
                progress(0.45)
            
            # Check for cancel after main processing
            if is_cancelled():
                return False, "ยกเลิกโดยผู้ใช้"
            
            progress(0.55)
            
            # Step 8: Enhance Chain (70%)
            log("🎛️ กำลังปรับแต่ง...")
            data = self._apply_enhance_chain(data, sr, config, log)
            progress(0.7)
            
            # Step 9: Truncate (80%)
            if config.truncate.enabled:
                data = self._apply_truncate(data, sr, config.truncate, log)
            progress(0.8)
            
            # Step 10: Apply Stereo Mode (90%)
            from constants import StereoMode
            stereo_mode = config.output.stereo_mode
            force_output = getattr(config.output, 'force_output_mode', False)
            
            # Normalize stereo_mode to string for comparison
            if isinstance(stereo_mode, StereoMode):
                stereo_mode_str = stereo_mode.value
            else:
                stereo_mode_str = str(stereo_mode)
            
            # Verbose logging for debugging stereo mode
            verbose = getattr(config.denoise, 'verbose_logging', False)
            if verbose:
                log(f"   [Verbose] stereo_mode={stereo_mode_str}, is_true_stereo={is_true_stereo}, force_output={force_output}, data.ndim={data.ndim}")
            
            if is_true_stereo and not force_output:
                # True stereo input - keep as stereo, ignore StereoMode setting
                log("   ✓ Output: Stereo (ตามต้นฉบับ)")
            else:
                # Apply StereoMode for mono input OR force mode
                if is_true_stereo and force_output:
                    # Mix stereo to mono first
                    log("   ⚠️ Force Output Mode: Mix Stereo → Mono")
                    data = np.mean(data, axis=1) if data.ndim == 2 else data
                
                if data.ndim == 1:  # Mono audio
                    if stereo_mode_str == "dual_mono":
                        data = self._apply_dual_mono(data, sr, log)
                    elif stereo_mode_str == "stereo_widening":
                        data = self._apply_mono_to_stereo(data, sr, log)
                    else:
                        log(f"   ✓ Output: Mono (stereo_mode={stereo_mode_str})")
                else:
                    log(f"   ⚠️ Data is 2D but expected 1D for stereo_mode")
            progress(0.9)
            
            # Step 11: Peak Normalize Output (95%)
            data = self._apply_peak_normalize(data, sr, log)
            progress(0.95)
            
            # Step 12: Save (100%)
            log(f"💾 กำลังบันทึก...")
            self._write_audio(data, sr, output_path, config.output.format)
            progress(1.0)
            
            return True, "สำเร็จ"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)
    
    def _read_audio(self, path: str) -> Tuple[np.ndarray, int, bool]:
        """
        อ่านไฟล์เสียงผ่าน FFmpeg
        Returns: (data, sr, is_true_stereo)
        - is_true_stereo = True if both channels have significant audio
        """
        creation_flags = 0x08000000 if os.name == 'nt' else 0
        
        command = [
            'ffmpeg', '-v', 'quiet',
            '-i', path,
            '-f', 'f32le',
            '-ac', '2',
            '-ar', str(self.sample_rate),
            '-'
        ]
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creation_flags
        )
        out, err = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg read error: {err.decode()}")
        
        raw_data = np.frombuffer(out, dtype=np.float32)
        stereo_data = raw_data.reshape(-1, 2)
        
        # Check if true stereo (both channels have significant audio)
        left_rms = np.sqrt(np.mean(stereo_data[:, 0] ** 2))
        right_rms = np.sqrt(np.mean(stereo_data[:, 1] ** 2))
        
        # If one channel has < 5% of the other's RMS, it's essentially mono
        if right_rms < left_rms * 0.05:
            # Right channel is silent, use left only
            return stereo_data[:, 0], self.sample_rate, False
        elif left_rms < right_rms * 0.05:
            # Left channel is silent, use right only
            return stereo_data[:, 1], self.sample_rate, False
        else:
            # True stereo
            return stereo_data, self.sample_rate, True
    
    def _fix_stereo_channel(
        self,
        data: np.ndarray,
        log: Callable[[str], None]
    ) -> np.ndarray:
        """แก้ไขปัญหาเสียงออกข้างเดียว (deprecated - handled in _read_audio)"""
        return data
    
    def _apply_silence_gate(
        self,
        data: np.ndarray,
        sr: int,
        threshold_db: float = -60.0
    ) -> np.ndarray:
        """
        Apply silence gate to convert low-level noise to true silence.
        
        STFT/iSTFT creates small artifacts at boundaries - this function
        gates them to true zeros so truncate can detect properly.
        
        Args:
            data: Audio data
            sr: Sample rate
            threshold_db: Below this level = silence (default -60dB)
        """
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Use short windows for detection (10ms)
        window_samples = int(sr * 0.01)
        if window_samples < 1:
            window_samples = 1
        
        # Process in windows
        n_windows = len(data) // window_samples
        if n_windows == 0:
            return data
        
        result = data.copy()
        
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            window = data[start:end]
            
            # If window max amplitude is below threshold, gate to zero
            if np.max(np.abs(window)) < threshold_linear:
                result[start:end] = 0.0
        
        # Handle leftover samples
        leftover_start = n_windows * window_samples
        if leftover_start < len(data):
            leftover = data[leftover_start:]
            if np.max(np.abs(leftover)) < threshold_linear:
                result[leftover_start:] = 0.0
        
        return result
    
    def _apply_dehum(
        self,
        data: np.ndarray,
        sr: int,
        log: Callable[[str], None]
    ) -> np.ndarray:
        """
        ลบเสียงจี่/Hum ด้วย Notch Filter
        ================================
        Hum (50/60Hz) เป็น stationary noise ที่ตัดด้วย notch filter ได้โดยตรง
        ไม่ต้องใช้ spectral subtraction ที่ซับซ้อน
        """
        log("   + ลบเสียงจี่ (Notch Filter)...")
        
        # Notch filters สำหรับ 50Hz hum และ harmonics
        # ใช้ Q สูงเพื่อตัดเฉพาะ frequency ที่ต้องการ ไม่กระทบเสียงพูด
        dehum_board = Pedalboard([
            PeakFilter(cutoff_frequency_hz=50, gain_db=-30, q=20),   # Fundamental
            PeakFilter(cutoff_frequency_hz=100, gain_db=-20, q=25),  # 2nd harmonic
            PeakFilter(cutoff_frequency_hz=150, gain_db=-15, q=30),  # 3rd harmonic
        ])
        
        result = dehum_board(data.astype(np.float32), sr)
        log("   ✓ ลบเสียงจี่เสร็จสิ้น")
        return result
    
    def _apply_lowcut(
        self,
        data: np.ndarray,
        sr: int,
        freq: float,
        log: Callable[[str], None]
    ) -> np.ndarray:
        """ตัดความถี่ต่ำ (ลดเสียงกระแทก, Rumble)"""
        log(f"   + Low-Cut: {freq}Hz")
        return Pedalboard([HighpassFilter(cutoff_frequency_hz=freq)])(data, sr)
    
    def _apply_transient_suppression(
        self,
        data: np.ndarray,
        sr: int,
        log: Callable[[str], None]
    ) -> np.ndarray:
        """
        ลดเสียงไมค์ช็อต / Click / Pop / Transient Spikes
        ==========================================
        ใช้ Median Filter + Spectral Repair
        ดีกว่า soft clipping เพราะรักษาคุณภาพเสียงได้ดีกว่า
        """
        log("⚡ กำลังลดเสียงไมค์ช็อต (Median Filter + Spectral Repair)...")
        
        # scipy imports already at module level
        
        # ===== ขั้นตอนที่ 1: ตรวจจับ Transients =====
        # คำนวณ local energy
        window_ms = 5  # 5ms window
        window_size = int(sr * window_ms / 1000)
        
        # คำนวณ short-term energy
        energy = np.convolve(data ** 2, np.ones(window_size) / window_size, mode='same')
        
        # คำนวณ derivative ของ energy เพื่อหา sudden changes
        energy_diff = np.abs(np.diff(energy, prepend=energy[0]))
        
        # หา threshold แบบ adaptive จาก median
        median_energy = np.median(energy_diff)
        std_energy = np.std(energy_diff)
        spike_threshold = median_energy + 4 * std_energy
        
        # หาตำแหน่งที่เป็น spike
        spike_mask = energy_diff > spike_threshold
        n_spikes = np.sum(spike_mask)
        
        if n_spikes == 0:
            log("   > ไม่พบ transient spikes")
            return data
        
        log(f"   > พบ {n_spikes} transient spikes")
        
        # ===== ขั้นตอนที่ 2: Median Filter สำหรับ spike regions =====
        # ใช้ median filter เฉพาะบริเวณที่มี spike
        # Median filter ดีสำหรับ impulse noise เพราะไม่ blur เสียงปกติ
        
        result = data.copy()
        
        # ขยาย spike mask ออกด้านละ 1ms
        dilation_size = int(sr * 0.001)
        dilated_mask = np.convolve(spike_mask.astype(float), 
                                   np.ones(dilation_size * 2), 
                                   mode='same') > 0
        
        # Apply median filter เฉพาะ spike regions
        median_window = 5  # samples (odd number)
        filtered = median_filter(data, size=median_window)
        
        # Replace spike regions with filtered version
        result[dilated_mask] = filtered[dilated_mask]
        
        # ===== ขั้นตอนที่ 3: Spectral Repair (optional smoothing) =====
        # ใช้ light lowpass filter เฉพาะ spike regions เพื่อ smooth ขอบ
        # แต่รักษาเสียงปกติไว้
        
        # หา boundaries ของ spike regions
        spike_starts = np.where(np.diff(dilated_mask.astype(int)) == 1)[0]
        spike_ends = np.where(np.diff(dilated_mask.astype(int)) == -1)[0]
        
        # Cross-fade ที่ขอบ spike regions
        fade_samples = int(sr * 0.001)  # 1ms fade
        for start, end in zip(spike_starts, spike_ends):
            # Fade in
            if start > fade_samples:
                fade_in = np.linspace(0, 1, fade_samples)
                result[start:start + fade_samples] = (
                    result[start:start + fade_samples] * fade_in +
                    data[start:start + fade_samples] * (1 - fade_in)
                )
            # Fade out
            if end + fade_samples < len(result):
                fade_out = np.linspace(1, 0, fade_samples)
                result[end:end + fade_samples] = (
                    result[end:end + fade_samples] * fade_out +
                    data[end:end + fade_samples] * (1 - fade_out)
                )
        
        log(f"   ✓ ซ่อมแซม {n_spikes} จุด เสร็จสิ้น")
        return result.astype(np.float32)
    
    def _apply_anti_drone(
        self,
        data: np.ndarray,
        sr: int,
        threshold: float,
        strength: float,
        log: Callable[[str], None]
    ) -> np.ndarray:
        """
        ลดเสียงลากยาว / Drone / Sustained Noise
        ใช้ noisereduce แบบ conservative เพื่อรักษาเสียงพูด
        
        Args:
            strength: 0.0-1.0 (higher = more aggressive noise reduction)
        """
        log(f"🔇 กำลังลดเสียงลากยาว (Anti-Drone, Thresh: {threshold:.1f}, Strength: {strength:.0%})...")
        
        # ใช้ noisereduce ด้วย stationary mode + fast settings
        # n_fft=2048 เร็วกว่า 4096 ประมาณ 2x แต่ยังจับ drone ได้ดี
        result = nr.reduce_noise(
            y=data,
            sr=sr,
            stationary=True,
            prop_decrease=strength,  # ใช้ค่าจาก user
            n_fft=2048,  # ลดจาก 4096 เพื่อเพิ่มความเร็ว
            n_std_thresh_stationary=threshold,  # ความไวในการตรวจจับ
            freq_mask_smooth_hz=250,  # smooth ใน frequency domain
            time_mask_smooth_ms=150   # smooth ใน time domain
        )
        
        # ใช้ filter อ่อนลงสำหรับ common drone frequencies
        drone_filter = Pedalboard([
            # ตัด very low frequencies ที่มักเป็น drone
            HighpassFilter(cutoff_frequency_hz=60),  # ลดจาก 80 เพื่อรักษา bass
            # Notch filter อ่อนลงสำหรับ common hum frequencies
            PeakFilter(cutoff_frequency_hz=60, gain_db=-6, q=3),   # ลดจาก -12dB
            PeakFilter(cutoff_frequency_hz=120, gain_db=-4, q=3),  # ลดจาก -8dB
        ])
        result = drone_filter(result, sr)
        
        log("   ✓ ลดเสียงลากยาวเสร็จสิ้น")
        return result
    
    def _apply_denoise(
        self,
        data: np.ndarray,
        sr: int,
        settings: 'DenoiseSettings',
        log: Callable[[str], None]
    ) -> np.ndarray:
        """
        ลด Noise ตาม Engine ที่เลือก
        - Engine: 8k (default), 16k, noisereduce
        """
        
        # ===== ตรวจสอบ Engine =====
        verbose = getattr(settings, 'verbose_logging', False)
        if verbose:
            log(f"   [Verbose] settings.engine = {settings.engine}")
        
        # Handle both Enum and string values
        engine_val = settings.engine.value if hasattr(settings.engine, 'value') else str(settings.engine)
        use_8k = (engine_val == "8k")
        use_16k = (engine_val == "16k")
        
        # ===== 8K Hybrid Engine (Default - Best for speech!) =====
        if use_8k:
            return self._apply_hybrid_8k(data, sr, settings, log)
        
        # ===== 16K Hybrid Engine =====
        if use_16k:
            return self._apply_hybrid_16k(data, sr, settings, log)
        
        # ===== noisereduce Engine (fallback) =====
        if settings.adaptive_enabled:
            log(f"🧹 Denoise (AI Mode - Strength {settings.strength:.2f})...")
            return nr.reduce_noise(
                y=data,
                sr=sr,
                stationary=False,
                prop_decrease=settings.strength,
                n_fft=settings.n_fft
            )
        else:
            log(f"🧹 Denoise (Standard - Strength {settings.strength:.2f})...")
            return nr.reduce_noise(
                y=data,
                sr=sr,
                stationary=True,
                prop_decrease=settings.strength,
                n_fft=settings.n_fft,
                n_std_thresh_stationary=1.5
            )
    
    def _apply_hybrid_8k(
        self,
        data: np.ndarray,
        sr: int,
        settings: 'DenoiseSettings',
        log: Callable[[str], None]
    ) -> np.ndarray:
        """
        ใช้ Hybrid 8kHz Engine - Best for speech!
        Native 8kHz processing for maximum speed.
        """
        log("🏆 Denoise (Hybrid 8K - Best for Speech!)...")
        
        try:
            # Temporarily add engine path for dependency resolution
            engine_dir = os.path.join(os.path.dirname(__file__), '8k')
            if engine_dir not in sys.path:
                sys.path.insert(0, engine_dir)
                path_added = True
            else:
                path_added = False
            
            try:
                from audio_processor_hybrid_8k import AudioProcessorHybrid8k
                
                processor = AudioProcessorHybrid8k()
                enhanced, info = processor.process(data, sr)
                
                # Apply strength blending
                strength = settings.strength
                if strength < 1.0:
                    result = strength * enhanced + (1 - strength) * data[:len(enhanced)]
                else:
                    result = enhanced
                
                log(f"   > Speed: {info.get('speed', 0):.1f}x realtime")
                log("   ✓ Hybrid 8K denoise completed")
                return result.astype(np.float32)
            finally:
                # Clean up path
                if path_added and engine_dir in sys.path:
                    sys.path.remove(engine_dir)
            
        except ImportError as e:
            log(f"   ❌ Hybrid 8K not available: {e}")
            log("   > Falling back to noisereduce...")
            return nr.reduce_noise(
                y=data, sr=sr, stationary=True,
                prop_decrease=settings.strength, n_fft=settings.n_fft
            )
        except Exception as e:
            log(f"   ❌ Hybrid 8K error: {e}")
            log("   > Falling back to noisereduce...")
            return nr.reduce_noise(
                y=data, sr=sr, stationary=True,
                prop_decrease=settings.strength, n_fft=settings.n_fft
            )
    
    def _apply_hybrid_16k(
        self,
        data: np.ndarray,
        sr: int,
        settings: 'DenoiseSettings',
        log: Callable[[str], None]
    ) -> np.ndarray:
        """
        ใช้ 16K V3 Engine (V12/V14/2-Pass)
        - V12: Digital Black + CRNN Blend
        - V14: V12 + HF Reduction 70%
        - 2-Pass: V7 → V12 (high noise)
        - 2-Pass + HF: V7 → V14
        """
        import math
        import time
        from scipy.signal import resample_poly
        
        # Get 16K mode from settings
        engine_mode = getattr(settings, 'engine_16k_mode', 'auto')
        if hasattr(engine_mode, 'value'):
            engine_mode = engine_mode.value
        
        # Log mode
        mode_names = {
            'auto': 'Auto (วิเคราะห์อัตโนมัติ)',
            'fast': 'Fast (V18 Smart)',
            'v12': 'V12 (Digital Black + CRNN)',
            'v14_hf': 'V14 (V12 + HF Reduction)',
            '2pass': '2-Pass (V7 → V12)',
            '2pass_hf': '2-Pass + HF (V7 → V14)'
        }
        log(f"🏆 Denoise (16K {mode_names.get(engine_mode, engine_mode)})...")
        
        try:
            # Temporarily add engine path for dependency resolution
            engine_dir = os.path.join(os.path.dirname(__file__), '16k_v3')
            if engine_dir not in sys.path:
                sys.path.insert(0, engine_dir)
                path_added = True
            else:
                path_added = False
            
            try:
                original_length = len(data)
                start_time = time.time()
                
                # Resample to 16kHz if needed
                if sr != 16000:
                    log(f"   > Resample {sr}Hz → 16kHz")
                    gcd = math.gcd(16000, sr)
                    audio_16k = resample_poly(data, 16000 // gcd, sr // gcd).astype(np.float32)
                else:
                    audio_16k = data.astype(np.float32)
                
                # ============ AUTO MODE ============
                if engine_mode == 'auto':
                    # Calculate noise floor for auto mode selection
                    def noise_floor_db(audio, sr_local=16000, percentile=5):
                        frame_size = int(0.03 * sr_local)
                        hop_size = frame_size // 2
                        n_frames = (len(audio) - frame_size) // hop_size
                        if n_frames <= 0:
                            return -60.0
                        rms_vals = np.array([
                            np.sqrt(np.mean(audio[i*hop_size:i*hop_size+frame_size]**2))
                            for i in range(n_frames)
                        ])
                        nf = np.percentile(rms_vals, percentile)
                        return 20 * np.log10(nf + 1e-10)
                    
                    nf = noise_floor_db(audio_16k)
                    log(f"   [AUTO] Noise Floor: {nf:.1f} dB")
                    
                    if nf > -30:
                        engine_mode = '2pass_hf'
                        log(f"   [AUTO] → 2-Pass + HF (noisy)")
                    elif nf > -35:
                        engine_mode = '2pass'
                        log(f"   [AUTO] → 2-Pass (moderate noise)")
                    else:
                        engine_mode = 'v14_hf'
                        log(f"   [AUTO] → V14 (clean)")
                
                # Select processor based on mode
                enhanced_16k = None
                
                if engine_mode == 'fast':
                    # V18 Fast Mode
                    log("   [V18] Smart processing...")
                    from audio_processor_v18_smart import AudioProcessorV18Smart
                    processor = AudioProcessorV18Smart()
                    enhanced_16k, info = processor.process(audio_16k, 16000)
                
                elif engine_mode in ['2pass', '2pass_hf']:
                    # 2-Pass Mode: V7 → V12/V14
                    log("   [Pass 1] V7 Pre-cleaning...")
                    from audio_processor_crnn_attention_latent import AudioProcessorHybrid16kNative as V7Processor
                    v7_processor = V7Processor()
                    pre_cleaned, _ = v7_processor.process(audio_16k, 16000)
                    
                    log("   [Pass 2] Final processing...")
                    if engine_mode == '2pass_hf':
                        from audio_processor_v14_combined import AudioProcessorV14Combined
                        final_processor = AudioProcessorV14Combined()
                    else:
                        from audio_processor_v12_blended import AudioProcessorV12Blended
                        final_processor = AudioProcessorV12Blended()
                    
                    enhanced_16k, info = final_processor.process(pre_cleaned, 16000)
                
                elif engine_mode == 'v14_hf':
                    # V14: V12 + HF Reduction
                    from audio_processor_v14_combined import AudioProcessorV14Combined
                    processor = AudioProcessorV14Combined()
                    enhanced_16k, info = processor.process(audio_16k, 16000)
                
                else:
                    # V12: Default (Digital Black + CRNN)
                    from audio_processor_v12_blended import AudioProcessorV12Blended
                    processor = AudioProcessorV12Blended()
                    enhanced_16k, info = processor.process(audio_16k, 16000)
                
                process_time = time.time() - start_time
                
                # Resample back to original sr if needed
                if sr != 16000:
                    log(f"   > Resample 16kHz → {sr}Hz")
                    gcd = math.gcd(sr, 16000)
                    enhanced = resample_poly(enhanced_16k, sr // gcd, 16000 // gcd).astype(np.float32)
                else:
                    enhanced = enhanced_16k
                
                # Match original length
                if len(enhanced) > original_length:
                    enhanced = enhanced[:original_length]
                elif len(enhanced) < original_length:
                    enhanced = np.pad(enhanced, (0, original_length - len(enhanced)))
                
                # Apply strength blending
                strength = settings.strength
                if strength < 1.0:
                    enhanced = strength * enhanced + (1 - strength) * data[:len(enhanced)]
                
                # Calculate speed
                audio_duration = len(data) / sr
                speed = audio_duration / process_time if process_time > 0 else 0
                
                log(f"   > Speed: {speed:.1f}x realtime")
                log(f"   ✓ 16K V3 ({engine_mode}) completed")
                return enhanced.astype(np.float32)
            finally:
                # Clean up path
                if path_added and engine_dir in sys.path:
                    sys.path.remove(engine_dir)
            
        except ImportError as e:
            log(f"   ❌ 16K V3 not available: {e}")
            log("   > Falling back to noisereduce...")
            return nr.reduce_noise(
                y=data, sr=sr, stationary=True,
                prop_decrease=settings.strength, n_fft=settings.n_fft
            )
        except Exception as e:
            log(f"   ❌ 16K V3 error: {e}")
            import traceback
            traceback.print_exc()
            log("   > Falling back to noisereduce...")
            return nr.reduce_noise(
                y=data, sr=sr, stationary=True,
                prop_decrease=settings.strength, n_fft=settings.n_fft
            )
    
    def _apply_wiener(
        self,
        data: np.ndarray,
        sr: int,
        settings: 'DenoiseSettings',
        log: Callable[[str], None]
    ) -> np.ndarray:
        """
        ใช้ Wiener Filter สำหรับ denoise
        Classic algorithm ที่เสถียรมาก ไม่มีปัญหา pitch เพี้ยน
        """
        from scipy.signal import wiener
        from scipy.ndimage import uniform_filter1d
        
        log("🎛️ Denoise (Wiener Filter - Classic)...")
        
        try:
            # ===== ขั้นตอนที่ 1: Estimate noise level =====
            # ใช้ local variance estimation
            window_size = int(sr * 0.02)  # 20ms window
            if window_size < 3:
                window_size = 3
            if window_size % 2 == 0:
                window_size += 1  # ต้องเป็นเลขคี่
            
            # Apply Wiener filter
            # mysize คือขนาด window สำหรับ local mean estimation
            filtered = wiener(data, mysize=window_size)
            
            # ===== ขั้นตอนที่ 2: Spectral subtraction เพิ่มเติม =====
            # ใช้ร่วมกับ Wiener เพื่อผลลัพธ์ที่ดีขึ้น
            from scipy import signal as scipy_signal
            
            # Compute STFT
            nperseg = 2048
            noverlap = nperseg // 2
            f, t, stft = scipy_signal.stft(filtered, sr, nperseg=nperseg, noverlap=noverlap)
            
            # Estimate noise from quiet frames
            mag = np.abs(stft)
            frame_power = np.mean(mag ** 2, axis=0)
            
            # หา 10% frame ที่เงียบที่สุด
            n_quiet = max(1, len(frame_power) // 10)
            quiet_indices = np.argsort(frame_power)[:n_quiet]
            noise_spectrum = np.mean(mag[:, quiet_indices], axis=1)
            
            # Spectral subtraction with soft thresholding
            alpha = 1.5 * settings.strength  # oversubtraction based on strength
            beta = 0.02  # spectral floor
            
            noise_broadcast = noise_spectrum[:, np.newaxis]
            clean_mag = mag - alpha * noise_broadcast
            clean_mag = np.maximum(clean_mag, beta * mag)
            
            # Reconstruct with original phase
            phase = np.angle(stft)
            clean_stft = clean_mag * np.exp(1j * phase)
            
            # Inverse STFT
            _, result = scipy_signal.istft(clean_stft, sr, nperseg=nperseg, noverlap=noverlap)
            
            # ===== ขั้นตอนที่ 3: Match length =====
            if len(result) > len(data):
                result = result[:len(data)]
            elif len(result) < len(data):
                result = np.pad(result, (0, len(data) - len(result)))
            
            # ===== ขั้นตอนที่ 4: Mix with original =====
            strength = settings.strength
            result = strength * result + (1 - strength) * data
            
            log("   ✓ Wiener filter completed")
            return result.astype(np.float32)
            
        except Exception as e:
            log(f"   ❌ Wiener error: {e}")
            log("   > Falling back to noisereduce...")
            return nr.reduce_noise(
                y=data,
                sr=sr,
                stationary=True,
                prop_decrease=settings.strength,
                n_fft=settings.n_fft
            )
    
    def _apply_silero(
        self,
        data: np.ndarray,
        sr: int,
        settings: 'DenoiseSettings',
        log: Callable[[str], None]
    ) -> np.ndarray:
        """
        ใช้ Silero Denoise - Offline mode supported
        โหลด model จาก ./silero/model/sns_latest.jit (ถ้ามี)
        Silero รับ 16kHz input และ output 48kHz
        """
        global SILERO_MODEL
        
        log("🧠 Denoise (Silero Neural Network)...")
        
        if not SILERO_AVAILABLE:
            log("   ❌ Silero not available (pip install torch torchaudio)")
            return data
        
        try:
            import torch
            import torchaudio
            
            # === โหลด Silero model ===
            if SILERO_MODEL is None:
                # ลองโหลดจาก local file ก่อน (offline mode)
                local_model_path = os.path.join(
                    os.path.dirname(__file__), 
                    "silero", "model", "sns_latest.jit"
                )
                
                if os.path.exists(local_model_path):
                    log(f"   > Loading local model: silero/model/sns_latest.jit")
                    torch._C._jit_set_profiling_mode(False)
                    torch.set_grad_enabled(False)
                    SILERO_MODEL = torch.jit.load(local_model_path)
                    SILERO_MODEL.eval()
                else:
                    # Fallback: โหลดจาก torch hub (ต้องมี internet)
                    log("   > Loading from torch hub (requires internet)...")
                    result = torch.hub.load(
                        repo_or_dir='snakers4/silero-models',
                        model='silero_denoise',
                        force_reload=False
                    )
                    SILERO_MODEL = result[0] if isinstance(result, tuple) else result
            
            model = SILERO_MODEL
            original_length = len(data)
            
            # === Resample to 16kHz (Silero input) ===
            if sr != 16000:
                log(f"   > Resample {sr}Hz -> 16kHz")
                resampler_to_16k = torchaudio.transforms.Resample(
                    orig_freq=sr, 
                    new_freq=16000
                )
                audio_16k = resampler_to_16k(torch.from_numpy(data.copy()).float())
            else:
                audio_16k = torch.from_numpy(data.copy()).float()
            
            # === Apply Silero (output = 48kHz) ===
            with torch.no_grad():
                enhanced_48k = model(audio_16k.unsqueeze(0)).squeeze(0)
            
            # === Resample back to original sr ===
            if sr != 48000:
                log(f"   > Resample 48kHz -> {sr}Hz")
                resampler_to_orig = torchaudio.transforms.Resample(
                    orig_freq=48000,
                    new_freq=sr
                )
                result = resampler_to_orig(enhanced_48k).numpy()
            else:
                result = enhanced_48k.numpy()
            
            # === Match length ===
            if len(result) > original_length:
                result = result[:original_length]
            elif len(result) < original_length:
                result = np.pad(result, (0, original_length - len(result)))
            
            # === Mix with original based on strength ===
            strength = settings.strength
            result = strength * result + (1 - strength) * data
            
            log("   ✓ Silero completed")
            return result.astype(np.float32)
            
        except Exception as e:
            log(f"   ❌ Silero error: {e}")
            import traceback
            traceback.print_exc()
            log("   > Falling back to noisereduce...")
            return nr.reduce_noise(
                y=data, sr=sr, stationary=True,
                prop_decrease=settings.strength, n_fft=settings.n_fft
            )
    
    def _apply_hybrid(
        self,
        data: np.ndarray,
        sr: int,
        settings: 'DenoiseSettings',
        log: Callable[[str], None]
    ) -> np.ndarray:
        """
        Hybrid Engine V3: noisereduce + Enhanced Wiener
        
        เร็วกว่า Silero มาก และไม่มีปัญหา memory
        ขั้นตอน:
        1. noisereduce (รักษาเสียงพูด)
        2. Enhanced Wiener (polish residual noise)
        """
        log("🔬 Denoise (Hybrid V3: noisereduce + Wiener)...")
        
        try:
            # === Stage 1: noisereduce (preserve speech) ===
            log("   [1/2] noisereduce (preserve speech)...")
            
            stage1 = nr.reduce_noise(
                y=data,
                sr=sr,
                stationary=settings.adaptive_enabled == False,
                prop_decrease=settings.strength * 0.85,  # เบาลงเล็กน้อย
                n_fft=settings.n_fft,
                n_std_thresh_stationary=1.2
            )
            
            # === Stage 2: Enhanced Wiener Polish ===
            log("   [2/2] Enhanced Wiener polish...")
            
            from scipy.signal import wiener
            from scipy import signal as scipy_signal
            
            # Wiener filter แบบ adaptive
            window_size = max(3, int(sr * 0.015))  # 15ms
            if window_size % 2 == 0:
                window_size += 1
            wiener_result = wiener(stage1, mysize=window_size)
            
            # Light spectral polish
            nperseg = min(2048, len(stage1) // 4)
            if nperseg > 64:
                f, t, stft = scipy_signal.stft(wiener_result, sr, nperseg=nperseg)
                mag = np.abs(stft)
                
                # Estimate noise floor from quietest 5%
                noise_floor = np.percentile(mag, 5, axis=1, keepdims=True)
                
                # Very light spectral subtraction
                alpha = 0.3 * settings.strength
                clean_mag = np.maximum(mag - alpha * noise_floor, 0.1 * mag)
                
                phase = np.angle(stft)
                clean_stft = clean_mag * np.exp(1j * phase)
                _, stage2 = scipy_signal.istft(clean_stft, sr, nperseg=nperseg)
                
                # Match length
                if len(stage2) > len(data):
                    stage2 = stage2[:len(data)]
                elif len(stage2) < len(data):
                    stage2 = np.pad(stage2, (0, len(data) - len(stage2)))
            else:
                stage2 = wiener_result
            
            # Final blend: 60% stage1 (noisereduce) + 40% stage2 (wiener)
            result = 0.6 * stage1 + 0.4 * stage2
            
            log("   ✓ Hybrid V3 completed")
            return result.astype(np.float32)
            
        except Exception as e:
            log(f"   ❌ Hybrid error: {e}")
            log("   > Falling back to noisereduce...")
            return nr.reduce_noise(
                y=data, sr=sr, stationary=True,
                prop_decrease=settings.strength, n_fft=settings.n_fft
            )
    
    def _apply_logmmse(
        self,
        data: np.ndarray,
        sr: int,
        settings: 'DenoiseSettings',
        log: Callable[[str], None]
    ) -> np.ndarray:
        """
        ใช้ H-Log-MMSE Speech Enhancement with AI-Adaptive Alpha
        =========================================================
        - AI-powered adaptive alpha per-frame (Tiny GRU)
        - Frequency-dependent Gain Floor (ป้องกัน musical noise)
        - Asymmetric Noise Tracking
        - +2.7 dB SNR improvement on real speech
        """
        try:
            from logmmse import AILogMMSEEnhancer, AI_AVAILABLE, LogMMSEEnhancer
            
            # Use AI version if available (has PyTorch + model)
            if AI_AVAILABLE:
                log("🧠 Denoise (H-Log-MMSE + AI Alpha)...")
                enhancer = AILogMMSEEnhancer(sample_rate=sr)
                if enhancer.use_ai:
                    log("   > AI adaptive alpha enabled")
                else:
                    log("   > Using fixed alpha (model not found)")
            else:
                log("🔬 Denoise (H-Log-MMSE)...")
                enhancer = LogMMSEEnhancer(sample_rate=sr)
            
            # Process
            enhanced = enhancer.process(data)
            
            # Apply strength blending
            strength = settings.strength
            if strength < 1.0:
                result = strength * enhanced + (1 - strength) * data[:len(enhanced)]
            else:
                result = enhanced
            
            log("   ✓ Log-MMSE denoise completed")
            return result.astype(np.float32)
            
        except ImportError as e:
            log(f"   ❌ Log-MMSE not available: {e}")
            log("   > Falling back to noisereduce...")
            return nr.reduce_noise(
                y=data, sr=sr, stationary=True,
                prop_decrease=settings.strength, n_fft=settings.n_fft
            )
        except Exception as e:
            log(f"   ❌ Log-MMSE error: {e}")
            log("   > Falling back to noisereduce...")
            return nr.reduce_noise(
                y=data, sr=sr, stationary=True,
                prop_decrease=settings.strength, n_fft=settings.n_fft
            )
    
    def _apply_omlsa(
        self,
        data: np.ndarray,
        sr: int,
        settings: 'DenoiseSettings',
        log: Callable[[str], None]
    ) -> np.ndarray:
        """
        ใช้ OM-LSA Speech Enhancement (Best Quality!)
        ==============================================
        - Optimally Modified Log-Spectral Amplitude (Cohen 2003)
        - AI-powered Speech Presence Probability (SPP)
        - Noise Freezing: ป้องกันการ estimate noise ผิดตอนมีเสียงพูด
        - +2.78 dB SNR improvement
        """
        log("🏆 Denoise (OM-LSA + AI SPP - Best Quality!)...")
        
        try:
            from omlsa import PresetModes
            
            # Get omlsa_mode from settings (default: balanced)
            omlsa_mode = getattr(settings, 'omlsa_mode', 'balanced')
            log(f"   > OM-LSA Mode: {omlsa_mode}")
            
            # Create enhancer based on mode
            if omlsa_mode == "conservative":
                enhancer = PresetModes.conservative()
            elif omlsa_mode == "aggressive":
                enhancer = PresetModes.aggressive()
            elif omlsa_mode == "protected":
                enhancer = PresetModes.protected()
            else:  # balanced (default)
                enhancer = PresetModes.balanced()
            
            # Override sample rate
            enhancer.sample_rate = sr
            
            if enhancer.use_ai:
                log("   > AI SPP + Alpha enabled")
            else:
                log("   > Using fallback mode")
            
            # Process
            enhanced = enhancer.process(data)
            
            # Apply strength blending
            strength = settings.strength
            if strength < 1.0:
                result = strength * enhanced + (1 - strength) * data[:len(enhanced)]
            else:
                result = enhanced
            
            log("   ✓ OM-LSA denoise completed")
            return result.astype(np.float32)
            
        except ImportError as e:
            log(f"   ❌ OM-LSA not available: {e}")
            log("   > Falling back to Log-MMSE...")
            return self._apply_logmmse(data, sr, settings, log)
        except Exception as e:
            log(f"   ❌ OM-LSA error: {e}")
            log("   > Falling back to noisereduce...")
            return nr.reduce_noise(
                y=data, sr=sr, stationary=True,
                prop_decrease=settings.strength, n_fft=settings.n_fft
            )
    
    def _apply_moe(
        self,
        data: np.ndarray,
        sr: int,
        settings: 'DenoiseSettings',
        log: Callable[[str], None]
    ) -> np.ndarray:
        """
        MoE Speech Enhancement (Best Quality!)
        =======================================
        - Multi-Expert system with 10 environment pre-filters
        - 6 Fixes: SPP Freezing, Clamped Oversub, Smart Floor, 
                   Loudness Comp, Temporal Smooth, SPP Gamma
        - Chunked processing with crossfade
        """
        log("🏆 Denoise (MoE + 6 Fixes - Best Quality!)...")
        
        try:
            from moe import IntegratedMoE
            
            # Create MoE enhancer
            moe = IntegratedMoE()
            log("   > MoE with 10 pre-filters loaded")
            
            # Process
            enhanced, chunk_info = moe.process(data, sr)
            
            # Log summary
            env_counts = {}
            for info in chunk_info:
                env = info["environment"]
                env_counts[env] = env_counts.get(env, 0) + 1
            
            if env_counts:
                top_env = max(env_counts, key=env_counts.get)
                log(f"   > Dominant environment: {top_env}")
            
            # Apply strength blending
            strength = settings.strength
            if strength < 1.0:
                result = strength * enhanced + (1 - strength) * data[:len(enhanced)]
            else:
                result = enhanced
            
            log("   ✓ MoE denoise completed")
            return result.astype(np.float32)
            
        except ImportError as e:
            log(f"   ❌ MoE not available: {e}")
            log("   > Falling back to OM-LSA...")
            return self._apply_omlsa(data, sr, settings, log)
        except Exception as e:
            log(f"   ❌ MoE error: {e}")
            log("   > Falling back to noisereduce...")
            return nr.reduce_noise(
                y=data, sr=sr, stationary=True,
                prop_decrease=settings.strength, n_fft=settings.n_fft
            )
    
    def _has_loudness_compensation(self, engine) -> bool:
        """
        Check if engine has built-in loudness compensation.
        Used to skip auto-gain for Hybrid engines.
        """
        from constants import DenoiseEngine
        
        if isinstance(engine, str):
            return engine in ["8k", "16k"]
        elif isinstance(engine, DenoiseEngine):
            return engine in [DenoiseEngine.HYBRID_8K, DenoiseEngine.HYBRID_16K]
        elif hasattr(engine, 'value'):
            return engine.value in ["8k", "16k"]
        return False
    
    def _apply_enhance_chain(
        self,
        data: np.ndarray,
        sr: int,
        config: ProcessingConfig,
        log: Callable[[str], None]
    ) -> np.ndarray:
        """ใช้ chain effects ทั้งหมด (Gain, EQ, Dynamics)"""
        effects = []
        
        # Check if engine has built-in loudness compensation
        is_hybrid_engine = self._has_loudness_compensation(config.denoise.engine)
        
        total_gain = config.output.manual_gain_db
        if config.output.normalize_enabled and not is_hybrid_engine:
            non_silent = np.abs(data) if data.ndim == 1 else np.abs(data).max(axis=1)
            if len(non_silent) > 0:
                speech_lvl = np.percentile(non_silent, 95)
                if speech_lvl > 0:
                    speech_db = 20 * np.log10(speech_lvl)
                    if speech_db < -6.0:
                        auto_gain = min(-6.0 - speech_db, 30.0)
                        total_gain += auto_gain
                        log(f"   + Auto Gain: +{auto_gain:.2f}dB")
        elif config.output.normalize_enabled and is_hybrid_engine:
            log("   ⏭️ Skip Auto Gain (8k/16k has built-in loudness compensation)")
        
        if total_gain != 0:
            effects.append(Gain(gain_db=total_gain))
        
        # EQ
        if config.eq.enabled:
            effects.append(LowShelfFilter(
                cutoff_frequency_hz=250,
                gain_db=config.eq.bass_gain
            ))
            effects.append(HighShelfFilter(
                cutoff_frequency_hz=3000,
                gain_db=config.eq.treble_gain
            ))
        
        # High Cut
        if config.eq.highcut_enabled:
            log(f"   + High-Cut: {config.eq.highcut_freq}Hz")
            effects.append(LowpassFilter(cutoff_frequency_hz=config.eq.highcut_freq))
        
        # De-esser
        if config.eq.deesser_enabled:
            log(f"   + De-esser: {config.eq.deesser_gain}dB @ {config.eq.deesser_freq}Hz")
            effects.append(PeakFilter(
                cutoff_frequency_hz=config.eq.deesser_freq,
                gain_db=config.eq.deesser_gain,
                q=1.0
            ))
        
        # Noise Gate
        if config.dynamics.gate_enabled:
            effects.append(NoiseGate(
                threshold_db=config.dynamics.gate_threshold,
                ratio=config.dynamics.gate_ratio,
                release_ms=config.dynamics.gate_release_ms
            ))
        
        # Compressor
        if config.dynamics.compressor_enabled:
            effects.append(Compressor(
                threshold_db=config.dynamics.compressor_threshold,
                ratio=config.dynamics.compressor_ratio,
                attack_ms=config.dynamics.compressor_attack_ms,
                release_ms=config.dynamics.compressor_release_ms
            ))
        
        # Limiter (safety wall)
        if config.dynamics.limiter_enabled:
            effects.append(Limiter(threshold_db=config.dynamics.limiter_threshold))
        
        if effects:
            return Pedalboard(effects)(data, sr)
        return data
    
    def _apply_truncate(
        self,
        data: np.ndarray,
        sr: int,
        settings: 'TruncateSettings',
        log: Callable[[str], None]
    ) -> np.ndarray:
        """ตัดช่วงเงียบออก (Smart Truncate) - รองรับทั้ง mono และ stereo"""
        log("✂️ กำลังตัดช่วงเงียบ...")
        
        if settings.min_silence_duration <= 0:
            return data
        
        # Handle stereo: use mono mix for analysis
        is_stereo = data.ndim == 2
        if is_stereo:
            analysis_data = np.mean(data, axis=1)  # Mix to mono for analysis
        else:
            analysis_data = data
        
        threshold_linear = 10 ** (settings.threshold_db / 20)
        chunk_size = int(sr * CHUNK_SIZE_SECONDS)
        
        if len(analysis_data) < chunk_size:
            return data
        
        # Pad to make divisible by chunk_size
        pad_len = chunk_size - (len(analysis_data) % chunk_size)
        if pad_len < chunk_size:
            working_data = np.pad(analysis_data, (0, pad_len))
        else:
            working_data = analysis_data
        
        windows = working_data.reshape(-1, chunk_size)
        max_amps = np.max(np.abs(windows), axis=1)
        is_silent = max_amps < threshold_linear
        
        keep_mask = np.ones(len(is_silent), dtype=bool)
        min_chunks = int(settings.min_silence_duration / CHUNK_SIZE_SECONDS)
        keep_chunks = int(settings.keep_silence / CHUNK_SIZE_SECONDS)
        
        i = 0
        while i < len(is_silent):
            if is_silent[i]:
                j = i
                while j < len(is_silent) and is_silent[j]:
                    j += 1
                if (j - i) > min_chunks:
                    keep_mask[i + keep_chunks:j] = False
                i = j
            else:
                i += 1
        
        # Apply mask to original data
        if is_stereo:
            # For stereo, we need to keep corresponding samples from both channels
            kept_chunks = np.where(keep_mask)[0]
            result_chunks = []
            for chunk_idx in kept_chunks:
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, len(data))
                result_chunks.append(data[start:end])
            
            if result_chunks:
                result = np.vstack(result_chunks)
            else:
                result = data
        else:
            result = windows[keep_mask].flatten()
            # Trim to original length if needed
            original_len = len(data)
            if len(result) > original_len:
                result = result[:original_len]
        
        removed_sec = (len(data) - len(result)) / sr
        log(f"   ✓ ตัดออก {removed_sec:.2f} วินาที")
        
        return result
    
    def _apply_peak_normalize(
        self,
        data: np.ndarray,
        sr: int,
        log: Callable[[str], None],
        target_db: float = -0.5
    ) -> np.ndarray:
        """
        Peak Normalize audio to target dB level
        ========================================
        ปรับ volume ให้ peak อยู่ที่ระดับที่กำหนด (default: -0.5 dB)
        """
        # Get current peak level
        if data.ndim == 1:
            current_peak = np.max(np.abs(data))
        else:
            current_peak = np.max(np.abs(data))
        
        if current_peak < 1e-10:
            return data
        
        # Calculate target level (linear)
        target_linear = 10 ** (target_db / 20)  # -0.5 dB ≈ 0.944
        
        # Calculate gain needed
        gain = target_linear / current_peak
        
        # Apply gain
        if gain > 1.0:  # Only boost if needed
            current_db = 20 * np.log10(current_peak + 1e-10)
            gain_db = 20 * np.log10(gain)
            log(f"📢 Peak Normalize: {current_db:.1f}dB → {target_db}dB (+{gain_db:.1f}dB)")
            data = data * gain
        
        return data.astype(np.float32)
    
    def _apply_dual_mono(
        self,
        data: np.ndarray,
        sr: int,
        log: Callable[[str], None]
    ) -> np.ndarray:
        """
        แปลง Mono เป็น Dual Mono (copy ทั้งสอง channel)
        ================================================
        """
        log("🔊 แปลง Mono → Dual Mono...")
        stereo = np.column_stack([data, data])
        log("   ✓ Dual mono applied")
        return stereo.astype(np.float32)
    
    def _apply_mono_to_stereo(
        self,
        data: np.ndarray,
        sr: int,
        log: Callable[[str], None]
    ) -> np.ndarray:
        """
        แปลง Mono เป็น Stereo พร้อม Widening Effect
        =============================================
        สร้างความกว้างโดยใช้:
        1. Slight delay บนช่อง R (Haas effect)
        2. High-frequency content difference
        3. Subtle low-frequency phase shift
        """
        log("🔊 แปลง Mono → Stereo (Widening)...")
        
        try:
            from scipy import signal as scipy_signal
            
            # === Parameters ===
            delay_ms = 8  # Haas effect delay (8-15ms ไม่ได้ยินเป็น echo)
            delay_samples = int(sr * delay_ms / 1000)
            width = 0.3  # Stereo width (0 = mono, 1 = wide)
            
            # === Left channel = original ===
            left = data.copy()
            
            # === Right channel = delayed + high-freq difference ===
            # 1. Add slight delay for Haas effect
            right = np.zeros_like(data)
            right[delay_samples:] = data[:-delay_samples]
            right[:delay_samples] = data[:delay_samples]
            
            # 2. Add high-frequency difference (create "air")
            # Use high-pass filter to extract high frequencies
            nyquist = sr / 2
            high_cutoff = 2000 / nyquist
            if high_cutoff < 1.0:
                b_high, a_high = scipy_signal.butter(2, high_cutoff, btype='high')
                high_freq = scipy_signal.filtfilt(b_high, a_high, data)
                
                # Add subtle difference
                left = left + width * 0.3 * high_freq
                right = right - width * 0.3 * high_freq
            
            # 3. Mid-side processing for extra width
            mid = (left + right) / 2
            side = (left - right) / 2
            
            # Enhance side signal slightly
            side = side * (1 + width * 0.5)
            
            # Reconstruct L/R
            left = mid + side
            right = mid - side
            
            # === Normalize to prevent clipping ===
            max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
            if max_val > 0.99:
                scale = 0.99 / max_val
                left = left * scale
                right = right * scale
            
            # === Stack to stereo (2D array) ===
            stereo = np.column_stack([left, right])
            
            log("   ✓ Stereo widening applied")
            return stereo.astype(np.float32)
            
        except Exception as e:
            log(f"   ⚠️ Stereo widening failed: {e}")
            log("   > Using dual mono...")
            # Fallback: simple dual mono
            return np.column_stack([data, data]).astype(np.float32)
    
    def _write_audio(
        self,
        data: np.ndarray,
        sr: int,
        path: str,
        format: str
    ) -> None:
        """บันทึกไฟล์เสียงผ่าน FFmpeg (รองรับ mono และ stereo)"""
        creation_flags = 0x08000000 if os.name == 'nt' else 0
        
        # Detect number of channels
        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]  # (samples, channels)
        
        command = [
            'ffmpeg', '-y',
            '-f', 'f32le',
            '-ar', str(sr),
            '-ac', str(channels),
            '-i', '-'
        ]
        
        # Add codec settings
        codec_settings = FFMPEG_CODECS.get(format, {})
        if 'codec' in codec_settings:
            command.extend(['-c:a', codec_settings['codec']])
        if 'bitrate_flag' in codec_settings:
            command.extend([codec_settings['bitrate_flag'], codec_settings['bitrate']])
        
        command.append(path)
        
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creation_flags
        )
        
        # Ensure data is in correct format for FFmpeg (interleaved if stereo)
        audio_data = data.astype(np.float32)
        if data.ndim == 2:
            # Interleave stereo: L R L R L R ...
            audio_data = audio_data.flatten('C')
        
        process.communicate(input=audio_data.tobytes())
        
        if process.returncode != 0:
            raise Exception("FFmpeg write error")
    
    def get_audio_info(self, path: str) -> Dict[str, Any]:
        """ดึงข้อมูลไฟล์เสียง"""
        creation_flags = 0x08000000 if os.name == 'nt' else 0
        
        command = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            path
        ]
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                creationflags=creation_flags
            )
            import json
            return json.loads(result.stdout)
        except:
            return {}
