"""
Audio Enhancer V2 - Constants and Enums
========================================
รวม Constants, Default Values, และ Enums ทั้งหมด
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any

# --- Version Info ---
APP_VERSION = "4.0.0"
APP_NAME = "Audio Studio AI Pro"
APP_TITLE = f"{APP_NAME} - V{APP_VERSION}"

# --- Supported Formats ---
SUPPORTED_AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma')
OUTPUT_FORMATS = ['mp3', 'wav', 'm4a', 'flac', 'ogg']

# --- Processing Defaults ---
DEFAULT_SAMPLE_RATE = 44100
CHUNK_SIZE_SECONDS = 0.05  # 50ms chunks for silence detection

# --- FFmpeg Codec Mapping ---
FFMPEG_CODECS = {
    'mp3': {'codec': 'libmp3lame', 'bitrate_flag': '-b:a', 'bitrate': '192k'},
    'm4a': {'codec': 'aac', 'bitrate_flag': '-b:a', 'bitrate': '192k'},
    'ogg': {'codec': 'libvorbis', 'bitrate_flag': '-q:a', 'bitrate': '5'},
    'wav': {'codec': 'pcm_s16le'},
    'flac': {'codec': 'flac'},
}


class ProcessingMode(Enum):
    """โหมดการ Denoise"""
    STATIONARY = "stationary"
    ADAPTIVE = "adaptive"
    ANTI_DRONE = "anti_drone"


class DenoiseEngine(Enum):
    """Engine สำหรับ Denoise"""
    HYBRID_8K = "8k"              # Default: 8kHz Hybrid (Best for speech!)
    HYBRID_16K = "16k"            # 16kHz Native Hybrid
    NOISEREDUCE = "noisereduce"   # Fallback: noisereduce library


class OMLSAMode(Enum):
    """OM-LSA Preset Modes"""
    BALANCED = "balanced"            # Balanced (default)
    CONSERVATIVE = "conservative"    # เน้น speech quality
    AGGRESSIVE = "aggressive"        # เน้นตัด noise
    PROTECTED = "protected"          # SPP-based speech protection


class Engine16kMode(Enum):
    """16K Engine Processing Mode"""
    AUTO = "auto"             # Auto-detect noise level and choose best mode
    FAST = "fast"             # V18 Smart - fast with auto V7 passes
    V12 = "v12"               # Digital Black + CRNN Blend (Default)
    V14_HF = "v14_hf"         # V12 + HF Reduction 70%
    TWO_PASS = "2pass"        # V7 → V12 (high noise)
    TWO_PASS_HF = "2pass_hf"  # V7 → V14 (high noise + HF)


class StereoMode(Enum):
    """Output Stereo Mode"""
    MONO = "mono"                    # Keep mono output
    DUAL_MONO = "dual_mono"          # Copy to both channels
    STEREO_WIDENING = "stereo_widening"  # Haas effect + mid-side


class Theme(Enum):
    """UI Theme"""
    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"


@dataclass
class DenoiseSettings:
    """การตั้งค่า Noise Reduction"""
    enabled: bool = True
    strength: float = 0.75
    mode: ProcessingMode = ProcessingMode.STATIONARY
    n_fft: int = 2048
    
    # Engine selection
    engine: DenoiseEngine = DenoiseEngine.HYBRID_8K
    
    # Adaptive mode
    adaptive_enabled: bool = False
    
    # Anti-drone mode
    anti_drone_enabled: bool = True
    anti_drone_threshold: float = 8.0
    anti_drone_strength: float = 0.75  # prop_decrease: 0.0-1.0 (higher = more aggressive)
    
    # Transient suppression (separate from anti-drone)
    transient_suppression_enabled: bool = True
    
    # Verbose logging (show detailed debug messages)
    verbose_logging: bool = False
    
    # OM-LSA preset mode (used when engine=OMLSA)
    omlsa_mode: str = "protected"  # balanced, conservative, aggressive, protected
    
    # 16K Engine mode (used when engine=HYBRID_16K)
    engine_16k_mode: str = "auto"  # auto, v12, v14_hf, 2pass, 2pass_hf


@dataclass
class DynamicsSettings:
    """การตั้งค่า Dynamics (Gate, Compressor, Limiter)"""
    # Low Cut
    lowcut_enabled: bool = True
    lowcut_freq: float = 100.0
    
    # Noise Gate
    gate_enabled: bool = True
    gate_threshold: float = -35.0
    gate_ratio: float = 10.0
    gate_release_ms: float = 250.0
    
    # Compressor
    compressor_enabled: bool = True
    compressor_threshold: float = -20.0
    compressor_ratio: float = 3.5
    compressor_attack_ms: float = 1.0
    compressor_release_ms: float = 100.0
    
    # Limiter
    limiter_enabled: bool = True
    limiter_threshold: float = -1.0


@dataclass
class EQSettings:
    """การตั้งค่า EQ และ Tone"""
    enabled: bool = False
    bass_gain: float = 0.0
    treble_gain: float = 0.0
    
    # De-Hum (50Hz removal)
    dehum_enabled: bool = True
    
    # High Cut
    highcut_enabled: bool = True
    highcut_freq: float = 10000.0
    
    # De-esser
    deesser_enabled: bool = True
    deesser_gain: float = -4.0
    deesser_freq: float = 7000.0


@dataclass
class TruncateSettings:
    """การตั้งค่า Smart Truncate"""
    enabled: bool = True
    threshold_db: float = -40.0
    min_silence_duration: float = 0.75
    keep_silence: float = 0.5


@dataclass
class OutputSettings:
    """การตั้งค่า Output"""
    format: str = 'mp3'
    folder: str = ''  # Empty = same as input
    normalize_enabled: bool = True
    fix_mono_enabled: bool = True
    manual_gain_db: float = 0.0
    stereo_mode: StereoMode = StereoMode.DUAL_MONO  # Output channel mode
    force_output_mode: bool = False  # Force apply stereo_mode even for true stereo input


@dataclass
class ProcessingConfig:
    """การตั้งค่าทั้งหมดรวมกัน"""
    denoise: DenoiseSettings = field(default_factory=DenoiseSettings)
    dynamics: DynamicsSettings = field(default_factory=DynamicsSettings)
    eq: EQSettings = field(default_factory=EQSettings)
    truncate: TruncateSettings = field(default_factory=TruncateSettings)
    output: OutputSettings = field(default_factory=OutputSettings)


# --- Default Presets ---
DEFAULT_PRESETS = {
    'default': {
        'name': 'Default',
        'description': 'การตั้งค่าเริ่มต้น สมดุลระหว่างคุณภาพและความเร็ว',
        'config': ProcessingConfig()
    },
    'podcast': {
        'name': 'Podcast',
        'description': 'เหมาะสำหรับ Podcast และเสียงพูด',
        'config': ProcessingConfig(
            denoise=DenoiseSettings(strength=0.75),
            dynamics=DynamicsSettings(compressor_ratio=4.0),
            eq=EQSettings(enabled=True, bass_gain=-2.0, treble_gain=2.0)
        )
    },
    'music': {
        'name': 'Music',
        'description': 'เหมาะสำหรับเพลง ใช้ denoise น้อยลง',
        'config': ProcessingConfig(
            denoise=DenoiseSettings(strength=0.4),
            dynamics=DynamicsSettings(compressor_ratio=2.5)
        )
    },
    'aggressive': {
        'name': 'Aggressive Clean',
        'description': 'ล้างเสียงรบกวนแรงสุด (อาจมี artifacts)',
        'config': ProcessingConfig(
            denoise=DenoiseSettings(strength=0.95, anti_drone_enabled=True),
            dynamics=DynamicsSettings(gate_threshold=-30.0)
        )
    }
}


# --- Language Strings ---
LANG = {
    'TH': {
        'app_title': f"{APP_NAME} - V{APP_VERSION}",
        'menu_file': "ไฟล์",
        'menu_settings': "ตั้งค่า",
        'menu_presets': "Presets",
        'menu_help': "ช่วยเหลือ",
        
        'tab_main': "หน้าหลัก",
        'tab_advanced': "ขั้นสูง",
        'tab_log': "บันทึก",
        
        'section_input': "ไฟล์ต้นฉบับ",
        'section_options': "ตัวเลือกการประมวลผล",
        'section_output': "ไฟล์ปลายทาง",
        
        'btn_add_files': "เพิ่มไฟล์",
        'btn_add_folder': "เพิ่มโฟลเดอร์",
        'btn_clear': "ล้างรายการ",
        'btn_browse': "เลือกโฟลเดอร์",
        'btn_start': "เริ่มประมวลผล",
        'btn_stop': "หยุด",
        'btn_open_folder': "เปิดโฟลเดอร์",
        
        'lbl_format': "รูปแบบไฟล์:",
        'lbl_output_folder': "โฟลเดอร์ปลายทาง:",
        'lbl_preset': "Preset:",
        'lbl_drop_hint': "ลากไฟล์มาวางที่นี่",
        
        'chk_normalize': "ปรับความดังอัตโนมัติ",
        'chk_denoise': "ลดเสียงรบกวน",
        'chk_fix_mono': "แก้เสียงออกข้างเดียว",
        'chk_truncate': "ตัดช่วงเงียบ",
        'chk_anti_drone': "ลดเสียงลากยาว",
        'chk_transient': "ลดเสียงไมค์ช็อต/Click",
        'chk_dehum': "ลดเสียงจี่ไฟ",
        'chk_adaptive': "โหมด AI (ช้าแต่แม่นยำ)",
        
        'status_ready': "พร้อมใช้งาน",
        'status_processing': "กำลังประมวลผล...",
        'status_done': "เสร็จสิ้น",
        'status_error': "เกิดข้อผิดพลาด",
        
        'msg_no_files': "กรุณาเพิ่มไฟล์ก่อน",
        'msg_no_ffmpeg': "ไม่พบ FFmpeg กรุณาติดตั้ง",
        'msg_success': "ประมวลผลสำเร็จ",
        'msg_failed': "ประมวลผลล้มเหลว",
        
        'theme_dark': "โหมดมืด",
        'theme_light': "โหมดสว่าง",
    },
    'EN': {
        'app_title': f"{APP_NAME} - V{APP_VERSION}",
        'menu_file': "File",
        'menu_settings': "Settings",
        'menu_presets': "Presets",
        'menu_help': "Help",
        
        'tab_main': "Main",
        'tab_advanced': "Advanced",
        'tab_log': "Log",
        
        'section_input': "Input Files",
        'section_options': "Processing Options",
        'section_output': "Output",
        
        'btn_add_files': "Add Files",
        'btn_add_folder': "Add Folder",
        'btn_clear': "Clear",
        'btn_browse': "Browse",
        'btn_start': "Start Processing",
        'btn_stop': "Stop",
        'btn_open_folder': "Open Folder",
        
        'lbl_format': "Format:",
        'lbl_output_folder': "Output Folder:",
        'lbl_preset': "Preset:",
        'lbl_drop_hint': "Drag & Drop files here",
        
        'chk_normalize': "Auto Normalize",
        'chk_denoise': "Noise Reduction",
        'chk_fix_mono': "Fix One-Sided Audio",
        'chk_truncate': "Smart Truncate",
        'chk_anti_drone': "Anti-Drone",
        'chk_transient': "Mic Shock/Transient",
        'chk_dehum': "Remove Hum",
        'chk_adaptive': "AI Mode (Slow but Accurate)",
        
        'status_ready': "Ready",
        'status_processing': "Processing...",
        'status_done': "Done",
        'status_error': "Error",
        
        'msg_no_files': "Please add files first",
        'msg_no_ffmpeg': "FFmpeg not found. Please install it.",
        'msg_success': "Processing Complete",
        'msg_failed': "Processing Failed",
        
        'theme_dark': "Dark Mode",
        'theme_light': "Light Mode",
    }
}


# --- Color Schemes ---
COLORS = {
    'dark': {
        'primary': '#3498db',
        'success': '#2ecc71',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'info': '#00d2d3',
        'bg': '#1e272e',
        'fg': '#ffffff',
        'log_bg': '#1e272e',
        'log_fg': '#00d2d3',
    },
    'light': {
        'primary': '#2980b9',
        'success': '#27ae60',
        'warning': '#e67e22',
        'danger': '#c0392b',
        'info': '#16a085',
        'bg': '#f5f6fa',
        'fg': '#2c3e50',
        'log_bg': '#ffffff',
        'log_fg': '#2c3e50',
    }
}
