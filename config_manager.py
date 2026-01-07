"""
Audio Enhancer V2 - Configuration Manager
==========================================
จัดการ Settings, Presets, และการ Migrate จาก version เก่า
"""

import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import asdict, fields
from pathlib import Path

from constants import (
    ProcessingConfig, DenoiseSettings, DynamicsSettings,
    EQSettings, TruncateSettings, OutputSettings,
    DEFAULT_PRESETS, Theme
)

# Config version - bump this when adding new fields
# v5: Added DenoiseEngine.MOE
# v6: Added transient_suppression_enabled (V3.0.2)
# v7: Added engine_16k_mode (V4.0.0)
CONFIG_VERSION = 7


class ConfigManager:
    """
    Configuration & Preset Manager
    ==============================
    - Load/Save user settings
    - Load/Save/List presets
    - Migrate old V1 settings
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Args:
            config_dir: directory สำหรับเก็บ config และ presets
                       (default: same as script)
        """
        if config_dir is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "settings_v2.json"
        self.presets_dir = self.config_dir / "presets"
        self.old_config_file = self.config_dir / "settings.json"
        
        # Ensure presets directory exists
        self.presets_dir.mkdir(exist_ok=True)
        
        # Initialize default presets if not exist
        self._init_default_presets()
    
    def _init_default_presets(self) -> None:
        """สร้าง preset เริ่มต้นถ้ายังไม่มี"""
        for preset_id, preset_data in DEFAULT_PRESETS.items():
            preset_file = self.presets_dir / f"{preset_id}.json"
            if not preset_file.exists():
                self.save_preset(
                    preset_id,
                    preset_data['name'],
                    preset_data['description'],
                    preset_data['config']
                )
    
    # ==================== User Settings ====================
    
    def load_config(self) -> Dict[str, Any]:
        """
        โหลด user settings
        
        Returns:
            Dict containing:
            - config: ProcessingConfig
            - theme: Theme
            - language: str ('TH' or 'EN')
            - window_geometry: str
            - last_output_folder: str
            - last_preset: str
        """
        default = {
            'config': ProcessingConfig(),
            'theme': Theme.DARK,
            'language': 'TH',
            'window_geometry': '900x750+100+0',  # position: +X+Y (top-left)
            'last_output_folder': '',
            'last_preset': 'default'
        }
        
        # Try new config first
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check config version - reset if outdated
                saved_version = data.get('version', 0)
                if saved_version < CONFIG_VERSION:
                    print(f"Config version outdated ({saved_version} < {CONFIG_VERSION}), resetting...")
                    return default
                
                return self._parse_config_data(data, default)
            except Exception as e:
                print(f"Error loading config: {e}")
        
        # Try migrating old config
        if self.old_config_file.exists():
            try:
                migrated = self._migrate_old_config()
                if migrated:
                    return migrated
            except Exception as e:
                print(f"Error migrating old config: {e}")
        
        return default
    
    def save_config(
        self,
        config: ProcessingConfig,
        theme: Theme,
        language: str,
        window_geometry: str,
        last_output_folder: str,
        last_preset: str
    ) -> bool:
        """บันทึก user settings"""
        try:
            data = {
                'version': CONFIG_VERSION,
                'config': self._config_to_dict(config),
                'theme': theme.value if isinstance(theme, Theme) else theme,
                'language': language,
                'window_geometry': window_geometry,
                'last_output_folder': last_output_folder,
                'last_preset': last_preset
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def _parse_config_data(
        self,
        data: Dict[str, Any],
        default: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse loaded JSON to proper types"""
        result = default.copy()
        
        if 'config' in data:
            result['config'] = self._dict_to_config(data['config'])
        
        if 'theme' in data:
            try:
                result['theme'] = Theme(data['theme'])
            except:
                result['theme'] = Theme.DARK
        
        result['language'] = data.get('language', 'TH')
        result['window_geometry'] = data.get('window_geometry', '900x750')
        result['last_output_folder'] = data.get('last_output_folder', '')
        result['last_preset'] = data.get('last_preset', 'default')
        
        return result
    
    # ==================== Presets ====================
    
    def list_presets(self) -> List[Dict[str, str]]:
        """
        รายการ presets ทั้งหมด
        
        Returns:
            List of dicts with 'id', 'name', 'description'
        """
        presets = []
        
        for preset_file in self.presets_dir.glob("*.json"):
            try:
                with open(preset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    presets.append({
                        'id': preset_file.stem,
                        'name': data.get('name', preset_file.stem),
                        'description': data.get('description', '')
                    })
            except:
                continue
        
        return presets
    
    def load_preset(self, preset_id: str) -> Optional[ProcessingConfig]:
        """
        โหลด preset
        
        Args:
            preset_id: ID ของ preset (ชื่อไฟล์ไม่รวม .json)
        
        Returns:
            ProcessingConfig หรือ None ถ้าไม่พบ
        """
        preset_file = self.presets_dir / f"{preset_id}.json"
        
        if not preset_file.exists():
            return None
        
        try:
            with open(preset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return self._dict_to_config(data.get('config', {}))
        except:
            return None
    
    def save_preset(
        self,
        preset_id: str,
        name: str,
        description: str,
        config: ProcessingConfig
    ) -> bool:
        """
        บันทึก preset
        
        Args:
            preset_id: ID ของ preset (จะเป็นชื่อไฟล์)
            name: ชื่อแสดงผล
            description: คำอธิบาย
            config: การตั้งค่า
        
        Returns:
            True ถ้าสำเร็จ
        """
        try:
            preset_file = self.presets_dir / f"{preset_id}.json"
            
            data = {
                'name': name,
                'description': description,
                'config': self._config_to_dict(config)
            }
            
            with open(preset_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving preset: {e}")
            return False
    
    def delete_preset(self, preset_id: str) -> bool:
        """ลบ preset (ยกเว้น built-in presets)"""
        if preset_id in DEFAULT_PRESETS:
            return False  # Cannot delete built-in presets
        
        preset_file = self.presets_dir / f"{preset_id}.json"
        try:
            if preset_file.exists():
                preset_file.unlink()
            return True
        except:
            return False
    
    # ==================== Migration ====================
    
    def _migrate_old_config(self) -> Optional[Dict[str, Any]]:
        """แปลง settings จาก V1 เป็น V2 format"""
        with open(self.old_config_file, 'r', encoding='utf-8') as f:
            old = json.load(f)
        
        # Map old settings to new structure
        config = ProcessingConfig(
            denoise=DenoiseSettings(
                enabled=old.get('denoise_enable', True),
                strength=old.get('denoise_strength', 0.65),
                adaptive_enabled=old.get('denoise_adaptive', False),
                anti_drone_enabled=old.get('antidrone_enable', False),
                anti_drone_threshold=old.get('antidrone_thresh', 3.5)
            ),
            dynamics=DynamicsSettings(
                lowcut_enabled=old.get('lowcut_enable', True),
                gate_enabled=old.get('gate_enable', True),
                gate_threshold=old.get('gate_thresh', -35.0),
                compressor_enabled=old.get('comp_enable', True),
                compressor_ratio=old.get('comp_ratio', 3.5),
                limiter_enabled=old.get('limiter_enable', True)
            ),
            eq=EQSettings(
                enabled=old.get('eq_enable', False),
                bass_gain=old.get('eq_bass', 0.0),
                treble_gain=old.get('eq_treble', 0.0),
                dehum_enabled=old.get('dehum_enable', False),
                highcut_enabled=old.get('highcut_enable', False),
                highcut_freq=old.get('highcut_freq', 10000.0),
                deesser_enabled=old.get('deess_enable', False),
                deesser_gain=old.get('deess_gain', -4.0)
            ),
            truncate=TruncateSettings(
                enabled=old.get('truncate_enable', False),
                threshold_db=old.get('trunc_thresh', -40.0),
                min_silence_duration=old.get('trunc_min_dur', 1.0),
                keep_silence=old.get('trunc_keep', 0.3)
            ),
            output=OutputSettings(
                format=old.get('output_format', 'mp3'),
                folder=old.get('custom_output_folder', ''),
                normalize_enabled=old.get('normalize_enable', True),
                fix_mono_enabled=old.get('chan_fix_enable', True),
                manual_gain_db=old.get('manual_gain', 0.0)
            )
        )
        
        result = {
            'config': config,
            'theme': Theme.DARK,
            'language': old.get('lang', 'TH'),
            'window_geometry': old.get('window_geometry', '900x750'),
            'last_output_folder': old.get('custom_output_folder', ''),
            'last_preset': 'default'
        }
        
        # Save migrated config
        self.save_config(
            config,
            result['theme'],
            result['language'],
            result['window_geometry'],
            result['last_output_folder'],
            result['last_preset']
        )
        
        return result
    
    # ==================== Helpers ====================
    
    def _serialize_value(self, value: Any) -> Any:
        """แปลงค่าให้เป็น JSON serializable"""
        from enum import Enum
        if isinstance(value, Enum):
            return value.value
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        return value
    
    def _config_to_dict(self, config: ProcessingConfig) -> Dict[str, Any]:
        """แปลง ProcessingConfig เป็น dict สำหรับ JSON"""
        result = {
            'denoise': asdict(config.denoise) if hasattr(config, 'denoise') else {},
            'dynamics': asdict(config.dynamics) if hasattr(config, 'dynamics') else {},
            'eq': asdict(config.eq) if hasattr(config, 'eq') else {},
            'truncate': asdict(config.truncate) if hasattr(config, 'truncate') else {},
            'output': asdict(config.output) if hasattr(config, 'output') else {}
        }
        # Serialize enums
        return self._serialize_value(result)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> ProcessingConfig:
        """แปลง dict กลับเป็น ProcessingConfig พร้อม migration logic"""
        
        def safe_create(cls, data_dict):
            """สร้าง dataclass จาก dict โดยกรอง field ที่ไม่รู้จักออก"""
            if not data_dict:
                return cls()
            valid_fields = {f.name for f in fields(cls)}
            filtered = {k: v for k, v in data_dict.items() if k in valid_fields}
            instance = cls(**filtered)
            
            # === V3.0.2 Migration ===
            # Add transient_suppression_enabled if missing (from old configs)
            if cls.__name__ == 'DenoiseSettings':
                if 'transient_suppression_enabled' not in data_dict:
                    instance.transient_suppression_enabled = True
                # === V4.0.0 Migration ===
                # Add engine_16k_mode if missing
                if 'engine_16k_mode' not in data_dict:
                    instance.engine_16k_mode = 'auto'
            
            return instance
        
        return ProcessingConfig(
            denoise=safe_create(DenoiseSettings, data.get('denoise', {})),
            dynamics=safe_create(DynamicsSettings, data.get('dynamics', {})),
            eq=safe_create(EQSettings, data.get('eq', {})),
            truncate=safe_create(TruncateSettings, data.get('truncate', {})),
            output=safe_create(OutputSettings, data.get('output', {}))
        )
