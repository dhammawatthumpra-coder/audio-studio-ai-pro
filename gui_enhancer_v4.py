"""
Audio Enhancer V2 - Modern GUI Application
============================================
GUI หลักสำหรับ Audio Enhancement ใช้ CustomTkinter
รองรับ Dark/Light Mode, Drag & Drop, Presets
"""

import os
import sys
import threading
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import traceback

# CustomTkinter สำหรับ Modern UI
try:
    import customtkinter as ctk
    from CTkMessagebox import CTkMessagebox
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False
    print("Warning: customtkinter not installed. Run: pip install customtkinter CTkMessagebox")

# Local imports
from constants import (
    APP_TITLE, APP_VERSION, LANG, COLORS, Theme,
    SUPPORTED_AUDIO_EXTENSIONS, OUTPUT_FORMATS,
    ProcessingConfig, DenoiseSettings, DynamicsSettings,
    EQSettings, TruncateSettings, OutputSettings
)
from config_manager import ConfigManager
from audio_processor import AudioProcessor


class FileItem:
    """แทนไฟล์หนึ่งไฟล์ใน list"""
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        self.status = "pending"  # pending, processing, done, error
        self.progress = 0.0
        self.error_msg = ""


class DragDropFrame(ctk.CTkFrame):
    """Frame รองรับ Drag & Drop"""
    
    def __init__(self, master, on_drop: Callable[[List[str]], None], **kwargs):
        super().__init__(master, **kwargs)
        self.on_drop = on_drop
        
        # Bind drag-drop events (ต้องใช้ tkinterdnd2 หรือ windnd)
        # สำหรับตอนนี้ใช้ button แทน
        self.drop_label = ctk.CTkLabel(
            self,
            text="🎵 คลิกเพื่อเลือกไฟล์",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.drop_label.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Make clickable
        self.drop_label.bind("<Button-1>", self._on_click)
        self.bind("<Button-1>", self._on_click)
    
    def _on_click(self, event):
        files = filedialog.askopenfilenames(
            filetypes=[("Audio Files", " ".join(f"*{ext}" for ext in SUPPORTED_AUDIO_EXTENSIONS))]
        )
        if files:
            self.on_drop(list(files))


class FileListItem(ctk.CTkFrame):
    """รายการไฟล์แต่ละรายการพร้อม progress bar"""
    
    def __init__(self, master, file_item: FileItem, on_remove: Callable, **kwargs):
        super().__init__(master, **kwargs)
        self.file_item = file_item
        self.on_remove = on_remove
        
        self.grid_columnconfigure(1, weight=1)
        
        # Status icon
        self.status_label = ctk.CTkLabel(self, text="🎵", width=30)
        self.status_label.grid(row=0, column=0, padx=(5, 0))
        
        # Filename
        self.name_label = ctk.CTkLabel(
            self,
            text=file_item.name,
            anchor="w",
            font=ctk.CTkFont(size=12)
        )
        self.name_label.grid(row=0, column=1, padx=5, sticky="ew")
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self, width=100, height=8)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=0, column=2, padx=5)
        
        # Remove button
        self.remove_btn = ctk.CTkButton(
            self,
            text="✕",
            width=30,
            height=25,
            fg_color="transparent",
            hover_color="#e74c3c",
            command=lambda: on_remove(file_item)
        )
        self.remove_btn.grid(row=0, column=3, padx=(0, 5))
    
    def update_status(self, status: str, progress: float = 0.0):
        """อัพเดทสถานะและ progress"""
        self.file_item.status = status
        self.file_item.progress = progress
        
        icons = {
            "pending": "🎵",
            "processing": "⏳",
            "done": "✅",
            "error": "❌"
        }
        self.status_label.configure(text=icons.get(status, "🎵"))
        self.progress_bar.set(progress)


class AdvancedSettingsWindow(ctk.CTkToplevel):
    """หน้าต่างตั้งค่าขั้นสูง"""
    
    def __init__(self, master, config: ProcessingConfig, lang: str, on_save: Callable):
        super().__init__(master)
        self.config = config
        self.L = LANG[lang]
        self.on_save = on_save
        self.value_labels = {}  # เก็บ label แสดงค่า
        
        self.title("Advanced Settings")
        self.geometry("550x750")
        self.resizable(False, False)
        
        # Tabs
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_denoise = self.tabview.add("Denoise")
        self.tab_dynamics = self.tabview.add("Dynamics")
        self.tab_eq = self.tabview.add("EQ & Tone")
        self.tab_truncate = self.tabview.add("Truncate")
        
        self._setup_denoise_tab()
        self._setup_dynamics_tab()
        self._setup_eq_tab()
        self._setup_truncate_tab()
        
        # Save button
        save_btn = ctk.CTkButton(
            self,
            text="Save Settings",
            command=self._save,
            fg_color="#27ae60",
            hover_color="#2ecc71"
        )
        save_btn.pack(pady=10)
    
    def _create_slider_with_label(
        self, 
        frame, 
        label_text: str, 
        slider_id: str,
        from_: float, 
        to: float, 
        value: float,
        steps: int = 20,
        format_str: str = "{:.1f}",
        unit: str = ""
    ):
        """สร้าง slider พร้อม label แสดงค่า"""
        # Container frame
        container = ctk.CTkFrame(frame, fg_color="transparent")
        container.pack(fill="x", pady=(10, 2))
        
        # Label และ Value อยู่ในบรรทัดเดียว
        ctk.CTkLabel(container, text=label_text).pack(side="left")
        
        value_label = ctk.CTkLabel(
            container, 
            text=f"{format_str.format(value)}{unit}",
            text_color="#3498db",
            font=ctk.CTkFont(weight="bold")
        )
        value_label.pack(side="right", padx=10)
        self.value_labels[slider_id] = value_label
        
        # Slider
        slider = ctk.CTkSlider(frame, from_=from_, to=to, number_of_steps=steps)
        slider.set(value)
        slider.pack(fill="x", padx=20)
        
        # Update callback
        def update_value(v):
            value_label.configure(text=f"{format_str.format(v)}{unit}")
        slider.configure(command=update_value)
        
        return slider
    
    def _setup_denoise_tab(self):
        frame = self.tab_denoise
        
        # Enable
        self.denoise_enabled = ctk.CTkSwitch(frame, text="Enable Denoise")
        self.denoise_enabled.pack(anchor="w", pady=5)
        if self.config.denoise.enabled:
            self.denoise_enabled.select()
        
        # ===== Engine Selection =====
        engine_frame = ctk.CTkFrame(frame, fg_color="transparent")
        engine_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(engine_frame, text="Engine:").pack(side="left")
        
        # Import DenoiseEngine
        from constants import DenoiseEngine
        
        engine_values = ["8k", "16k", "noisereduce"]
        self.engine_menu = ctk.CTkOptionMenu(
            engine_frame,
            values=engine_values,
            width=150
        )
        # Get current engine value (may be Enum or string from JSON)
        engine_attr = getattr(self.config.denoise, 'engine', '8k')
        if hasattr(engine_attr, 'value'):
            current_engine = engine_attr.value
        else:
            current_engine = str(engine_attr) if engine_attr else "8k"
        # Handle old engine values - fallback to 8k
        if current_engine not in engine_values:
            current_engine = "8k"
        self.engine_menu.set(current_engine)
        self.engine_menu.pack(side="left", padx=10)
        
        # Status indicators - 16K Fast is recommended
        status_text = "🏆 16K Fast (V18) recommended for speech"
        status_color = "#27ae60"
        
        self.engine_status = ctk.CTkLabel(
            engine_frame, 
            text=status_text,
            text_color=status_color,
            font=ctk.CTkFont(size=10)
        )
        self.engine_status.pack(side="left", padx=10)
        
        # ===== 16K Mode Selection (show when 16k engine selected) =====
        self.mode_16k_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.mode_16k_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(self.mode_16k_frame, text="16K Mode:").pack(side="left")
        
        # Display labels (user-friendly) mapped to internal values - Language aware
        mode_16k_internal = ["auto", "fast", "v12", "v14_hf", "2pass", "2pass_hf"]
        mode_16k_labels = {
            'TH': ["อัตโนมัติ", "เร็ว (V18)", "มาตรฐาน", "ลดเสียงซ่า", "Noise เยอะ", "Noise+ซ่า"],
            'EN': ["Auto", "Fast (V18)", "Standard", "Reduce Hiss", "High Noise", "Noise+Hiss"]
        }
        # Get current language from parent
        current_lang = getattr(self, 'L', {})
        lang_key = 'TH' if 'ลาก' in str(current_lang.get('lbl_drop_hint', '')) else 'EN'
        mode_16k_display = mode_16k_labels.get(lang_key, mode_16k_labels['EN'])
        
        self._mode_16k_map = dict(zip(mode_16k_display, mode_16k_internal))
        self._mode_16k_reverse = dict(zip(mode_16k_internal, mode_16k_display))
        
        self.mode_16k_menu = ctk.CTkOptionMenu(
            self.mode_16k_frame,
            values=mode_16k_display,
            width=180
        )
        current_16k_mode = getattr(self.config.denoise, 'engine_16k_mode', 'auto')
        if hasattr(current_16k_mode, 'value'):
            current_16k_mode = current_16k_mode.value
        if current_16k_mode not in mode_16k_internal:
            current_16k_mode = "auto"
        # Set display label from internal value
        default_label = mode_16k_display[0] if mode_16k_display else "Auto"
        self.mode_16k_menu.set(self._mode_16k_reverse.get(current_16k_mode, default_label))
        self.mode_16k_menu.pack(side="left", padx=10)
        
        # Initially hide if not 16k engine
        if current_engine != "16k":
            self.mode_16k_frame.pack_forget()
        
        # Bind engine change to toggle 16k mode visibility
        def on_engine_change(value):
            if value == "16k":
                self.mode_16k_frame.pack(fill="x", pady=5, after=engine_frame)
            else:
                self.mode_16k_frame.pack_forget()
        
        self.engine_menu.configure(command=on_engine_change)
        
        # ===== Stereo Mode =====
        from constants import StereoMode
        
        stereo_frame = ctk.CTkFrame(frame, fg_color="transparent")
        stereo_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(stereo_frame, text="Output Mode:").pack(side="left")
        stereo_values = ["mono", "dual_mono", "stereo_widening"]
        stereo_labels = {"mono": "Mono", "dual_mono": "Dual Mono", "stereo_widening": "Stereo Widening"}
        
        self.stereo_mode_menu = ctk.CTkOptionMenu(
            stereo_frame,
            values=stereo_values,
            width=150
        )
        # Handle both str and StereoMode enum
        stereo_mode = self.config.output.stereo_mode
        current_stereo = stereo_mode.value if hasattr(stereo_mode, 'value') else str(stereo_mode)
        self.stereo_mode_menu.set(current_stereo)
        self.stereo_mode_menu.pack(side="left", padx=10)
        
        # ===== Force Output Mode =====
        self.force_output_mode = ctk.CTkSwitch(frame, text="Force Output Mode (บังคับใช้ Output Mode กับ Stereo)")
        self.force_output_mode.pack(anchor="w", pady=5)
        if getattr(self.config.output, 'force_output_mode', False):
            self.force_output_mode.select()
        
        # ===== AI Mode =====
        ctk.CTkLabel(frame, text="").pack()  # spacer
        self.adaptive_mode = ctk.CTkSwitch(frame, text="AI Mode (Slow but Accurate)")
        self.adaptive_mode.pack(anchor="w", padx=20, pady=5)
        if self.config.denoise.adaptive_enabled:
            self.adaptive_mode.select()
        
        # Strength
        self.denoise_strength = self._create_slider_with_label(
            frame, "Denoise Strength:", "denoise_strength",
            from_=0, to=1, value=self.config.denoise.strength,
            steps=20, format_str="{:.2f}"
        )
        
        # ===== Anti-Drone =====
        ctk.CTkLabel(frame, text="").pack()  # spacer
        self.anti_drone = ctk.CTkSwitch(frame, text="Anti-Drone Mode (Aggressive)")
        self.anti_drone.pack(anchor="w", pady=5)
        if self.config.denoise.anti_drone_enabled:
            self.anti_drone.select()
        
        self.anti_drone_thresh = self._create_slider_with_label(
            frame, "Anti-Drone Threshold:", "anti_drone_thresh",
            from_=1, to=10, value=self.config.denoise.anti_drone_threshold,
            steps=18, format_str="{:.1f}"
        )
        
        # Anti-Drone Strength (prop_decrease)
        default_strength = getattr(self.config.denoise, 'anti_drone_strength', 0.75)
        self.anti_drone_strength = self._create_slider_with_label(
            frame, "Anti-Drone Strength:", "anti_drone_strength",
            from_=0.3, to=1.0, value=default_strength,
            steps=14, format_str="{:.0%}"
        )
        
        # ===== Verbose Logging =====
        ctk.CTkLabel(frame, text="").pack()  # spacer
        self.verbose_logging = ctk.CTkSwitch(frame, text="Verbose Logging (Show Debug Info)")
        self.verbose_logging.pack(anchor="w", pady=5)
        if getattr(self.config.denoise, 'verbose_logging', False):
            self.verbose_logging.select()
    
    def _setup_dynamics_tab(self):
        frame = self.tab_dynamics
        
        # Low Cut
        self.lowcut_enabled = ctk.CTkSwitch(frame, text="Low-Cut Filter (Remove Rumble)")
        self.lowcut_enabled.pack(anchor="w", pady=5)
        if self.config.dynamics.lowcut_enabled:
            self.lowcut_enabled.select()
        
        self.lowcut_freq = self._create_slider_with_label(
            frame, "Low-Cut Frequency:", "lowcut_freq",
            from_=50, to=200, value=self.config.dynamics.lowcut_freq,
            steps=15, format_str="{:.0f}", unit=" Hz"
        )
        
        # Noise Gate
        ctk.CTkLabel(frame, text="").pack()
        self.gate_enabled = ctk.CTkSwitch(frame, text="Noise Gate")
        self.gate_enabled.pack(anchor="w", pady=5)
        if self.config.dynamics.gate_enabled:
            self.gate_enabled.select()
        
        self.gate_thresh = self._create_slider_with_label(
            frame, "Gate Threshold:", "gate_thresh",
            from_=-60, to=-10, value=self.config.dynamics.gate_threshold,
            steps=50, format_str="{:.0f}", unit=" dB"
        )
        
        # Compressor
        ctk.CTkLabel(frame, text="").pack()
        self.comp_enabled = ctk.CTkSwitch(frame, text="Compressor")
        self.comp_enabled.pack(anchor="w", pady=5)
        if self.config.dynamics.compressor_enabled:
            self.comp_enabled.select()
        
        self.comp_ratio = self._create_slider_with_label(
            frame, "Compression Ratio:", "comp_ratio",
            from_=1, to=10, value=self.config.dynamics.compressor_ratio,
            steps=18, format_str="{:.1f}", unit=":1"
        )
        
        # Limiter
        ctk.CTkLabel(frame, text="").pack()
        self.limiter_enabled = ctk.CTkSwitch(frame, text="Limiter (Safety)")
        self.limiter_enabled.pack(anchor="w", pady=5)
        if self.config.dynamics.limiter_enabled:
            self.limiter_enabled.select()
    
    def _setup_eq_tab(self):
        frame = self.tab_eq
        
        # EQ Enable
        self.eq_enabled = ctk.CTkSwitch(frame, text="Enable EQ")
        self.eq_enabled.pack(anchor="w", pady=5)
        if self.config.eq.enabled:
            self.eq_enabled.select()
        
        self.bass_gain = self._create_slider_with_label(
            frame, "Bass Gain:", "bass_gain",
            from_=-10, to=10, value=self.config.eq.bass_gain,
            steps=20, format_str="{:+.0f}", unit=" dB"
        )
        
        self.treble_gain = self._create_slider_with_label(
            frame, "Treble Gain:", "treble_gain",
            from_=-10, to=10, value=self.config.eq.treble_gain,
            steps=20, format_str="{:+.0f}", unit=" dB"
        )
        
        # De-Hum
        ctk.CTkLabel(frame, text="").pack()
        self.dehum_enabled = ctk.CTkSwitch(frame, text="Remove Hum (50/60Hz)")
        self.dehum_enabled.pack(anchor="w", pady=5)
        if self.config.eq.dehum_enabled:
            self.dehum_enabled.select()
        
        # High Cut
        self.highcut_enabled = ctk.CTkSwitch(frame, text="High-Cut Filter")
        self.highcut_enabled.pack(anchor="w", pady=5)
        if self.config.eq.highcut_enabled:
            self.highcut_enabled.select()
        
        self.highcut_freq = self._create_slider_with_label(
            frame, "High-Cut Frequency:", "highcut_freq",
            from_=5000, to=15000, value=self.config.eq.highcut_freq,
            steps=20, format_str="{:.0f}", unit=" Hz"
        )
        
        # De-esser
        self.deesser_enabled = ctk.CTkSwitch(frame, text="De-esser")
        self.deesser_enabled.pack(anchor="w", pady=(15, 5))
        if self.config.eq.deesser_enabled:
            self.deesser_enabled.select()
    
    def _setup_truncate_tab(self):
        frame = self.tab_truncate
        
        self.truncate_enabled = ctk.CTkSwitch(frame, text="Enable Smart Truncate")
        self.truncate_enabled.pack(anchor="w", pady=5)
        if self.config.truncate.enabled:
            self.truncate_enabled.select()
        
        self.trunc_thresh = self._create_slider_with_label(
            frame, "Silence Threshold:", "trunc_thresh",
            from_=-60, to=-20, value=self.config.truncate.threshold_db,
            steps=40, format_str="{:.0f}", unit=" dB"
        )
        
        self.trunc_min_dur = self._create_slider_with_label(
            frame, "Min Silence Duration:", "trunc_min_dur",
            from_=0.5, to=5, value=self.config.truncate.min_silence_duration,
            steps=18, format_str="{:.1f}", unit=" sec"
        )
        
        self.trunc_keep = self._create_slider_with_label(
            frame, "Keep Silence:", "trunc_keep",
            from_=0.1, to=1, value=self.config.truncate.keep_silence,
            steps=18, format_str="{:.2f}", unit=" sec"
        )
    
    def _save(self):
        """บันทึกการตั้งค่า"""
        from constants import DenoiseEngine, StereoMode
        
        # Denoise settings
        self.config.denoise.enabled = self.denoise_enabled.get()
        self.config.denoise.adaptive_enabled = self.adaptive_mode.get()
        self.config.denoise.strength = self.denoise_strength.get()
        self.config.denoise.anti_drone_enabled = self.anti_drone.get()
        self.config.denoise.anti_drone_threshold = self.anti_drone_thresh.get()
        self.config.denoise.anti_drone_strength = self.anti_drone_strength.get()
        self.config.denoise.verbose_logging = self.verbose_logging.get()
        
        # Engine selection
        engine_value = self.engine_menu.get()
        self.config.denoise.engine = DenoiseEngine(engine_value)
        
        # 16K Mode selection (save if exists)
        if hasattr(self, 'mode_16k_menu') and hasattr(self, '_mode_16k_map'):
            display_value = self.mode_16k_menu.get()
            internal_value = self._mode_16k_map.get(display_value, 'v12')
            self.config.denoise.engine_16k_mode = internal_value
        
        # Stereo mode selection
        stereo_value = self.stereo_mode_menu.get()
        self.config.output.stereo_mode = StereoMode(stereo_value)
        
        # Force output mode
        self.config.output.force_output_mode = self.force_output_mode.get()
        
        self.config.dynamics.lowcut_enabled = self.lowcut_enabled.get()
        self.config.dynamics.lowcut_freq = self.lowcut_freq.get()
        self.config.dynamics.gate_enabled = self.gate_enabled.get()
        self.config.dynamics.gate_threshold = self.gate_thresh.get()
        self.config.dynamics.compressor_enabled = self.comp_enabled.get()
        self.config.dynamics.compressor_ratio = self.comp_ratio.get()
        self.config.dynamics.limiter_enabled = self.limiter_enabled.get()
        
        self.config.eq.enabled = self.eq_enabled.get()
        self.config.eq.bass_gain = self.bass_gain.get()
        self.config.eq.treble_gain = self.treble_gain.get()
        self.config.eq.dehum_enabled = self.dehum_enabled.get()
        self.config.eq.highcut_enabled = self.highcut_enabled.get()
        self.config.eq.highcut_freq = self.highcut_freq.get()
        self.config.eq.deesser_enabled = self.deesser_enabled.get()
        
        self.config.truncate.enabled = self.truncate_enabled.get()
        self.config.truncate.threshold_db = self.trunc_thresh.get()
        self.config.truncate.min_silence_duration = self.trunc_min_dur.get()
        self.config.truncate.keep_silence = self.trunc_keep.get()
        
        self.on_save(self.config)
        self.destroy()


class AudioEnhancerApp(ctk.CTk):
    """Main Application"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize managers
        self.config_manager = ConfigManager()
        self.processor = AudioProcessor()
        
        # Load config
        saved = self.config_manager.load_config()
        self.config: ProcessingConfig = saved['config']
        self.current_theme: Theme = saved['theme']
        self.current_lang: str = saved['language']
        self.last_output_folder: str = saved['last_output_folder']
        self.current_preset: str = saved['last_preset']
        
        # Set theme
        ctk.set_appearance_mode(self.current_theme.value)
        ctk.set_default_color_theme("blue")
        
        # Files list
        self.files: List[FileItem] = []
        self.file_widgets: dict = {}
        self.is_processing = False
        self.cancel_requested = False  # Cancel flag for stopping processing
        
        # Window setup
        self.title(LANG[self.current_lang]['app_title'])
        self.minsize(800, 650)
        
        # Set window icon (cross-platform)
        import platform
        icon_base = os.path.join(os.path.dirname(__file__), 'icon')
        if platform.system() == 'Windows':
            icon_path = icon_base + '.ico'
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
        elif platform.system() == 'Darwin':  # macOS
            # macOS uses .icns but Tkinter doesn't support it directly
            # Icon is typically set via .app bundle
            pass
        else:  # Linux
            icon_path = icon_base + '.png'
            if os.path.exists(icon_path):
                try:
                    img = tk.PhotoImage(file=icon_path)
                    self.iconphoto(True, img)
                except:
                    pass
        
        # Maximize window on startup (cross-platform)
        def maximize_window():
            if platform.system() == 'Windows':
                self.state('zoomed')
            elif platform.system() == 'Darwin':  # macOS
                self.attributes('-zoomed', True)
            else:  # Linux
                self.attributes('-zoomed', True)
        
        maximize_window()
        
        # Setup UI
        self._setup_ui()
        self._update_ui_text()
        
        # Re-maximize after UI is ready (fix for window shrinking)
        self.after(100, maximize_window)
        
        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _setup_ui(self):
        """สร้าง UI ทั้งหมด"""
        # Top bar (Theme, Language, Presets)
        self._setup_topbar()
        
        # Main content
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Left panel (Files)
        self._setup_file_panel()
        
        # Right panel (Options)
        self._setup_options_panel()
        
        # Bottom (Start button, Log)
        self._setup_bottom_panel()
    
    def _setup_topbar(self):
        """สร้าง top bar"""
        topbar = ctk.CTkFrame(self, height=50)
        topbar.pack(fill="x", padx=15, pady=15)
        
        # Theme toggle
        self.theme_switch = ctk.CTkSwitch(
            topbar,
            text="🌙 Dark",
            command=self._toggle_theme,
            onvalue=1,
            offvalue=0
        )
        self.theme_switch.pack(side="left", padx=10)
        if self.current_theme == Theme.DARK:
            self.theme_switch.select()
        
        # Language selector
        self.lang_menu = ctk.CTkSegmentedButton(
            topbar,
            values=["TH", "EN"],
            command=self._change_language
        )
        self.lang_menu.set(self.current_lang)
        self.lang_menu.pack(side="left", padx=10)
        
        # Preset selector
        ctk.CTkLabel(topbar, text="Preset:").pack(side="left", padx=(20, 5))
        
        presets = self.config_manager.list_presets()
        preset_names = [p['id'] for p in presets]
        
        self.preset_menu = ctk.CTkOptionMenu(
            topbar,
            values=preset_names,
            command=self._load_preset
        )
        self.preset_menu.set(self.current_preset)
        self.preset_menu.pack(side="left", padx=5)
        
        # Save preset button
        self.save_preset_btn = ctk.CTkButton(
            topbar,
            text="💾",
            width=35,
            command=self._save_preset_dialog
        )
        self.save_preset_btn.pack(side="left", padx=5)
    
    def _setup_file_panel(self):
        """สร้าง panel สำหรับไฟล์"""
        self.file_frame = ctk.CTkFrame(self.main_frame)
        self.file_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Header
        header = ctk.CTkFrame(self.file_frame, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=10)
        
        self.file_header_label = ctk.CTkLabel(
            header,
            text="📁 Input Files",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.file_header_label.pack(side="left")
        
        # Buttons
        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right")
        
        self.add_files_btn = ctk.CTkButton(
            btn_frame,
            text="+ Add",
            width=70,
            command=self._add_files
        )
        self.add_files_btn.pack(side="left", padx=2)
        
        self.add_folder_btn = ctk.CTkButton(
            btn_frame,
            text="📂",
            width=35,
            command=self._add_folder
        )
        self.add_folder_btn.pack(side="left", padx=2)
        
        self.clear_btn = ctk.CTkButton(
            btn_frame,
            text="Clear",
            width=60,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            command=self._clear_files
        )
        self.clear_btn.pack(side="left", padx=2)
        
        # File list (scrollable)
        self.file_list_frame = ctk.CTkScrollableFrame(self.file_frame)
        self.file_list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Drop zone (shown when empty)
        self.drop_zone = DragDropFrame(
            self.file_list_frame,
            on_drop=self._on_files_dropped,
            height=150
        )
        self.drop_zone.pack(fill="x", pady=5)
    
    def _setup_options_panel(self):
        """สร้าง panel สำหรับ options"""
        self.options_frame = ctk.CTkFrame(self.main_frame, width=280)
        self.options_frame.pack(side="right", fill="y")
        self.options_frame.pack_propagate(False)
        
        # Header
        self.options_header_label = ctk.CTkLabel(
            self.options_frame,
            text="⚙️ Options",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.options_header_label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Quick toggles
        toggles_frame = ctk.CTkFrame(self.options_frame, fg_color="transparent")
        toggles_frame.pack(fill="x", padx=15)
        
        self.normalize_switch = ctk.CTkSwitch(toggles_frame, text="Auto Normalize")
        self.normalize_switch.pack(anchor="w", pady=3)
        if self.config.output.normalize_enabled:
            self.normalize_switch.select()
        
        self.denoise_switch = ctk.CTkSwitch(toggles_frame, text="Noise Reduction")
        self.denoise_switch.pack(anchor="w", pady=3)
        if self.config.denoise.enabled:
            self.denoise_switch.select()
        
        self.fix_mono_switch = ctk.CTkSwitch(toggles_frame, text="Fix One-Sided Audio")
        self.fix_mono_switch.pack(anchor="w", pady=3)
        if self.config.output.fix_mono_enabled:
            self.fix_mono_switch.select()
        
        self.anti_drone_switch = ctk.CTkSwitch(toggles_frame, text="Anti-Drone Mode")
        self.anti_drone_switch.pack(anchor="w", pady=3)
        if self.config.denoise.anti_drone_enabled:
            self.anti_drone_switch.select()
        
        self.transient_switch = ctk.CTkSwitch(toggles_frame, text="Mic Shock/Transient")
        self.transient_switch.pack(anchor="w", pady=3)
        if getattr(self.config.denoise, 'transient_suppression_enabled', True):
            self.transient_switch.select()
        
        self.truncate_switch = ctk.CTkSwitch(toggles_frame, text="Smart Truncate")
        self.truncate_switch.pack(anchor="w", pady=3)
        if self.config.truncate.enabled:
            self.truncate_switch.select()
        
        self.dehum_switch = ctk.CTkSwitch(toggles_frame, text="Remove Hum")
        self.dehum_switch.pack(anchor="w", pady=3)
        if self.config.eq.dehum_enabled:
            self.dehum_switch.select()
        
        self.post_cleanup_switch = ctk.CTkSwitch(toggles_frame, text="Post-Process Cleanup")
        self.post_cleanup_switch.pack(anchor="w", pady=3)
        if getattr(self.config.denoise, 'post_cleanup_enabled', False):
            self.post_cleanup_switch.select()
        
        # Advanced button
        self.advanced_btn = ctk.CTkButton(
            self.options_frame,
            text="🔧 Advanced Settings",
            command=self._open_advanced_settings
        )
        self.advanced_btn.pack(fill="x", padx=15, pady=(15, 10))
        
        # Output section
        ctk.CTkLabel(
            self.options_frame,
            text="Output",
            font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=15, pady=(10, 5))
        
        # Format
        format_frame = ctk.CTkFrame(self.options_frame, fg_color="transparent")
        format_frame.pack(fill="x", padx=15)
        
        self.format_label = ctk.CTkLabel(format_frame, text="Format:")
        self.format_label.pack(side="left")
        
        self.format_menu = ctk.CTkOptionMenu(
            format_frame,
            values=OUTPUT_FORMATS,
            width=100
        )
        self.format_menu.set(self.config.output.format)
        self.format_menu.pack(side="right")
        
        # Output folder
        self.folder_label = ctk.CTkLabel(
            self.options_frame,
            text="Folder:",
            anchor="w"
        )
        self.folder_label.pack(anchor="w", padx=15, pady=(10, 2))
        
        folder_frame = ctk.CTkFrame(self.options_frame, fg_color="transparent")
        folder_frame.pack(fill="x", padx=15)
        
        self.folder_entry = ctk.CTkEntry(folder_frame)
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        if self.last_output_folder:
            self.folder_entry.insert(0, self.last_output_folder)
        else:
            self.folder_entry.insert(0, "[Same as Input]")
        
        self.browse_btn = ctk.CTkButton(
            folder_frame,
            text="...",
            width=35,
            command=self._browse_output_folder
        )
        self.browse_btn.pack(side="right")
    
    def _setup_bottom_panel(self):
        """สร้าง bottom panel (Start button, Log)"""
        bottom = ctk.CTkFrame(self)
        bottom.pack(fill="x", padx=15, pady=(0, 15))
        
        # Button frame for Start/Cancel
        btn_frame = ctk.CTkFrame(bottom, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(0, 5))
        
        # Start button
        self.start_btn = ctk.CTkButton(
            btn_frame,
            text="🚀 Start Processing",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            fg_color="#27ae60",
            hover_color="#2ecc71",
            command=self._start_processing
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        # Cancel button
        self.cancel_btn = ctk.CTkButton(
            btn_frame,
            text="⏹️ Cancel",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            width=100,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            command=self._cancel_processing,
            state="disabled"
        )
        self.cancel_btn.pack(side="right")
        
        # Progress bar (overall)
        self.overall_progress = ctk.CTkProgressBar(bottom)
        self.overall_progress.set(0)
        self.overall_progress.pack(fill="x", pady=(0, 5))
        
        # Log header with copy button
        log_header = ctk.CTkFrame(bottom, fg_color="transparent")
        log_header.pack(fill="x", pady=(0, 2))
        
        ctk.CTkLabel(log_header, text="📋 Log:").pack(side="left")
        
        self.copy_log_btn = ctk.CTkButton(
            log_header,
            text="📋 Copy",
            width=70,
            height=24,
            command=self._copy_log
        )
        self.copy_log_btn.pack(side="right")
        
        # Log area
        self.log_text = ctk.CTkTextbox(bottom, height=100, state="disabled")
        self.log_text.pack(fill="x")
        
        # Right-click context menu for log
        self.log_text.bind("<Button-3>", self._show_log_context_menu)
        
        # Status bar
        status_frame = ctk.CTkFrame(bottom, fg_color="transparent")
        status_frame.pack(fill="x", pady=(5, 0))
        
        self.status_label = ctk.CTkLabel(status_frame, text="Ready", text_color="gray")
        self.status_label.pack(side="left")
        
        self.open_folder_btn = ctk.CTkButton(
            status_frame,
            text="📂 Open Output Folder",
            width=150,
            command=self._open_output_folder
        )
        self.open_folder_btn.pack(side="right")
        self.open_folder_btn.pack_forget()  # Hide initially
    
    def _update_ui_text(self):
        """อัพเดทข้อความตามภาษา"""
        L = LANG[self.current_lang]
        
        self.title(L['app_title'])
        self.file_header_label.configure(text=f"📁 {L['section_input']}")
        self.options_header_label.configure(text=f"⚙️ {L['section_options']}")
        
        self.add_files_btn.configure(text=L['btn_add_files'])
        self.clear_btn.configure(text=L['btn_clear'])
        
        self.normalize_switch.configure(text=L['chk_normalize'])
        self.denoise_switch.configure(text=L['chk_denoise'])
        self.fix_mono_switch.configure(text=L['chk_fix_mono'])
        self.anti_drone_switch.configure(text=L['chk_anti_drone'])
        self.transient_switch.configure(text=L.get('chk_transient', 'Mic Shock/Transient'))
        self.truncate_switch.configure(text=L['chk_truncate'])
        self.dehum_switch.configure(text=L['chk_dehum'])
        
        self.format_label.configure(text=L['lbl_format'])
        self.drop_zone.drop_label.configure(text=L.get('lbl_drop_hint', 'Click to Select Files'))
        self.start_btn.configure(text=f"🚀 {L['btn_start']}")
        self.open_folder_btn.configure(text=f"📂 {L['btn_open_folder']}")
        self.status_label.configure(text=L['status_ready'])
    
    # ==================== Event Handlers ====================
    
    def _toggle_theme(self):
        """สลับ theme Dark/Light"""
        if self.theme_switch.get():
            self.current_theme = Theme.DARK
            self.theme_switch.configure(text="🌙 Dark")
        else:
            self.current_theme = Theme.LIGHT
            self.theme_switch.configure(text="☀️ Light")
        
        ctk.set_appearance_mode(self.current_theme.value)
        self._save_config()
    
    def _change_language(self, lang: str):
        """เปลี่ยนภาษา"""
        self.current_lang = lang
        self._update_ui_text()
        self._save_config()
    
    def _load_preset(self, preset_id: str):
        """โหลด preset"""
        config = self.config_manager.load_preset(preset_id)
        if config:
            self.config = config
            self.current_preset = preset_id
            self._refresh_switches()
            self._log(f"Loaded preset: {preset_id}")
            self._save_config()
    
    def _refresh_switches(self):
        """Refresh UI switches ตาม config"""
        if self.config.output.normalize_enabled:
            self.normalize_switch.select()
        else:
            self.normalize_switch.deselect()
        
        if self.config.denoise.enabled:
            self.denoise_switch.select()
        else:
            self.denoise_switch.deselect()
        
        if self.config.output.fix_mono_enabled:
            self.fix_mono_switch.select()
        else:
            self.fix_mono_switch.deselect()
        
        if self.config.denoise.anti_drone_enabled:
            self.anti_drone_switch.select()
        else:
            self.anti_drone_switch.deselect()
        
        if self.config.truncate.enabled:
            self.truncate_switch.select()
        else:
            self.truncate_switch.deselect()
        
        if getattr(self.config.denoise, 'transient_suppression_enabled', True):
            self.transient_switch.select()
        else:
            self.transient_switch.deselect()
        
        if self.config.eq.dehum_enabled:
            self.dehum_switch.select()
        else:
            self.dehum_switch.deselect()
        
        if getattr(self.config.denoise, 'post_cleanup_enabled', False):
            self.post_cleanup_switch.select()
        else:
            self.post_cleanup_switch.deselect()
        
        self.format_menu.set(self.config.output.format)
    
    def _save_preset_dialog(self):
        """แสดง dialog สำหรับบันทึก preset"""
        dialog = ctk.CTkInputDialog(
            text="Enter preset name:",
            title="Save Preset"
        )
        name = dialog.get_input()
        
        if name:
            preset_id = name.lower().replace(" ", "_")
            self._sync_config_from_ui()
            
            self.config_manager.save_preset(
                preset_id,
                name,
                "User preset",
                self.config
            )
            
            # Refresh preset menu
            presets = self.config_manager.list_presets()
            self.preset_menu.configure(values=[p['id'] for p in presets])
            self.preset_menu.set(preset_id)
            
            self._log(f"Saved preset: {name}")
    
    def _add_files(self):
        """เพิ่มไฟล์"""
        files = filedialog.askopenfilenames(
            filetypes=[("Audio Files", " ".join(f"*{ext}" for ext in SUPPORTED_AUDIO_EXTENSIONS))]
        )
        if files:
            self._on_files_dropped(list(files))
    
    def _add_folder(self):
        """เพิ่มโฟลเดอร์"""
        folder = filedialog.askdirectory()
        if folder:
            files = []
            for f in os.listdir(folder):
                if f.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS):
                    files.append(os.path.join(folder, f))
            if files:
                self._on_files_dropped(files)
    
    def _on_files_dropped(self, paths: List[str]):
        """เมื่อมีไฟล์ถูก drop หรือเลือก"""
        # Hide drop zone
        self.drop_zone.pack_forget()
        
        for path in paths:
            if os.path.abspath(path) not in [f.path for f in self.files]:
                file_item = FileItem(path)
                self.files.append(file_item)
                
                # Create widget
                widget = FileListItem(
                    self.file_list_frame,
                    file_item,
                    on_remove=self._remove_file
                )
                widget.pack(fill="x", pady=2)
                self.file_widgets[file_item.path] = widget
        
        self._log(f"Added {len(paths)} file(s)")
    
    def _remove_file(self, file_item: FileItem):
        """ลบไฟล์ออกจากรายการ"""
        if file_item.path in self.file_widgets:
            self.file_widgets[file_item.path].destroy()
            del self.file_widgets[file_item.path]
        
        self.files = [f for f in self.files if f.path != file_item.path]
        
        if not self.files:
            self.drop_zone.pack(fill="x", pady=5)
    
    def _clear_files(self):
        """ล้างรายการไฟล์ทั้งหมด"""
        for widget in self.file_widgets.values():
            widget.destroy()
        
        self.file_widgets.clear()
        self.files.clear()
        self.drop_zone.pack(fill="x", pady=5)
        self._log("Cleared all files")
    
    def _browse_output_folder(self):
        """เลือกโฟลเดอร์ปลายทาง"""
        folder = filedialog.askdirectory()
        if folder:
            self.last_output_folder = folder
            self.folder_entry.delete(0, "end")
            self.folder_entry.insert(0, folder)
            self._save_config()
    
    def _open_advanced_settings(self):
        """เปิดหน้าต่าง Advanced Settings"""
        self._sync_config_from_ui()
        
        window = AdvancedSettingsWindow(
            self,
            self.config,
            self.current_lang,
            on_save=self._on_advanced_save
        )
        window.grab_set()
    
    def _on_advanced_save(self, config: ProcessingConfig):
        """เมื่อบันทึกการตั้งค่าขั้นสูง"""
        self.config = config
        self._refresh_switches()
        self._save_config()
        self._log("Advanced settings saved")
    
    def _sync_config_from_ui(self):
        """Sync config จาก UI switches"""
        self.config.output.normalize_enabled = bool(self.normalize_switch.get())
        self.config.denoise.enabled = bool(self.denoise_switch.get())
        self.config.output.fix_mono_enabled = bool(self.fix_mono_switch.get())
        self.config.denoise.anti_drone_enabled = bool(self.anti_drone_switch.get())
        self.config.denoise.transient_suppression_enabled = bool(self.transient_switch.get())
        self.config.truncate.enabled = bool(self.truncate_switch.get())
        self.config.eq.dehum_enabled = bool(self.dehum_switch.get())
        self.config.denoise.post_cleanup_enabled = bool(self.post_cleanup_switch.get())
        self.config.output.format = self.format_menu.get()
    
    def _start_processing(self):
        """เริ่มการประมวลผล"""
        if not self.files:
            CTkMessagebox(
                title="No Files",
                message=LANG[self.current_lang]['msg_no_files'],
                icon="warning"
            )
            return
        
        if not self.processor.ffmpeg_available:
            CTkMessagebox(
                title="FFmpeg Missing",
                message=LANG[self.current_lang]['msg_no_ffmpeg'],
                icon="cancel"
            )
            return
        
        self._sync_config_from_ui()
        self.is_processing = True
        self.cancel_requested = False  # Reset cancel flag
        
        # Disable/Enable UI
        self.start_btn.configure(
            text="⏳ Processing...",
            state="disabled",
            fg_color="gray"
        )
        self.cancel_btn.configure(state="normal")  # Enable cancel button
        
        # Start processing thread
        thread = threading.Thread(target=self._process_files, daemon=True)
        thread.start()
    
    def _process_files(self):
        """ประมวลผลไฟล์ทั้งหมด (รันใน thread)"""
        total = len(self.files)
        success = 0
        
        self._log(f"--- เริ่มประมวลผล {total} ไฟล์ ---")
        
        for i, file_item in enumerate(self.files):
            # Check for cancel request
            if self.cancel_requested:
                self._log("⚠️ ยกเลิกตามคำขอ")
                break
            
            if not self.is_processing:
                break
            
            # Update widget status
            widget = self.file_widgets.get(file_item.path)
            if widget:
                self.after(0, lambda w=widget: w.update_status("processing", 0))
            
            # Determine output path
            output_folder = self.last_output_folder or os.path.dirname(file_item.path)
            base_name = os.path.splitext(file_item.name)[0]
            output_path = os.path.join(
                output_folder,
                f"{base_name}_enhanced.{self.config.output.format}"
            )
            
            # Process
            def progress_callback(p):
                if widget:
                    self.after(0, lambda: widget.update_status("processing", p))
            
            success_flag, msg = self.processor.process_file(
                file_item.path,
                output_path,
                self.config,
                log_func=self._log,
                progress_func=progress_callback,
                cancel_check=lambda: self.cancel_requested
            )
            
            if success_flag:
                success += 1
                if widget:
                    self.after(0, lambda w=widget: w.update_status("done", 1.0))
                self._log(f"✅ เสร็จสิ้น: {file_item.name}")
            else:
                if widget:
                    self.after(0, lambda w=widget: w.update_status("error", 0))
                self._log(f"❌ ล้มเหลว: {file_item.name} - {msg}")
            
            # Update overall progress
            overall = (i + 1) / total
            self.after(0, lambda p=overall: self.overall_progress.set(p))
        
        # Finish
        self._log(f"--- เสร็จสิ้น {success}/{total} ไฟล์ ---")
        
        self.after(0, self._on_processing_done)
    
    def _on_processing_done(self):
        """เมื่อประมวลผลเสร็จ"""
        self.is_processing = False
        
        L = LANG[self.current_lang]
        self.start_btn.configure(
            text=f"🚀 {L['btn_start']}",
            state="normal",
            fg_color="#27ae60"
        )
        self.cancel_btn.configure(state="disabled")  # Disable cancel button
        
        # Show different message if cancelled
        if self.cancel_requested:
            self.status_label.configure(text="Cancelled")
            CTkMessagebox(
                title="Cancelled",
                message="Processing was cancelled",
                icon="info"
            )
        else:
            self.status_label.configure(text=L['status_done'])
            self.open_folder_btn.pack(side="right")
            
            CTkMessagebox(
                title=L['msg_success'],
                message=f"Processed {len(self.files)} files",
                icon="check"
            )
    
    def _cancel_processing(self):
        """ยกเลิกการประมวลผล"""
        if self.is_processing:
            self.cancel_requested = True
            self._log("⚠️ กำลังยกเลิก... กรุณารอสักครู่")
            self.cancel_btn.configure(state="disabled")
    
    def _open_output_folder(self):
        """เปิดโฟลเดอร์ปลายทาง"""
        folder = self.last_output_folder
        if not folder and self.files:
            folder = os.path.dirname(self.files[0].path)
        
        if folder and os.path.exists(folder):
            if os.name == 'nt':
                os.startfile(folder)
            else:
                import subprocess
                subprocess.Popen(['xdg-open', folder])
    
    def _log(self, message: str):
        """เพิ่มข้อความใน log"""
        def update():
            self.log_text.configure(state="normal")
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.log_text.insert("end", f"[{timestamp}] {message}\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
            
            self.status_label.configure(text=message[:50])
        
        self.after(0, update)
    
    def _copy_log(self):
        """Copy log content to clipboard"""
        self.log_text.configure(state="normal")
        content = self.log_text.get("1.0", "end-1c")
        self.log_text.configure(state="disabled")
        
        self.clipboard_clear()
        self.clipboard_append(content)
        
        # Visual feedback
        original_text = self.copy_log_btn.cget("text")
        self.copy_log_btn.configure(text="✓ Copied!")
        self.after(1500, lambda: self.copy_log_btn.configure(text=original_text))
    
    def _show_log_context_menu(self, event):
        """Show context menu on right-click"""
        import tkinter as tk
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="📋 Copy All", command=self._copy_log)
        menu.add_command(label="🗑️ Clear Log", command=self._clear_log)
        menu.tk_popup(event.x_root, event.y_root)
    
    def _clear_log(self):
        """Clear log content"""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
    
    def _save_config(self):
        """บันทึก config"""
        self._sync_config_from_ui()
        self.config_manager.save_config(
            self.config,
            self.current_theme,
            self.current_lang,
            self.geometry(),
            self.last_output_folder,
            self.current_preset
        )
    
    def _on_close(self):
        """เมื่อปิดโปรแกรม"""
        self._save_config()
        self.destroy()


def main():
    """Entry point"""
    if not CTK_AVAILABLE:
        # Fallback message
        import tkinter.messagebox as mb
        mb.showerror(
            "Missing Dependencies",
            "Please install required packages:\n\n"
            "pip install customtkinter CTkMessagebox"
        )
        return
    
    app = AudioEnhancerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
