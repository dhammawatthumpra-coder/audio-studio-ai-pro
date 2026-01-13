# -*- mode: python ; coding: utf-8 -*-
# Audio Enhancer V3 - PyInstaller Spec File
# Build command: pyinstaller build.spec

import os
import sys

block_cipher = None

# Get current directory
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))

# Site-packages path (adjust if needed)
SITE_PACKAGES = r'C:\Users\ACER\miniconda3\Lib\site-packages'

# Collect all V4 files
added_files = [
    # Core directories
    ('8k', '8k'),
    ('16k_v3', '16k_v3'),
    ('presets', 'presets'),
    
    # Assets
    ('icon.ico', '.'),
    
    # CustomTkinter (entire package)
    (os.path.join(SITE_PACKAGES, 'customtkinter'), 'customtkinter'),
    (os.path.join(SITE_PACKAGES, 'CTkMessagebox'), 'CTkMessagebox'),
]

# Add Numba cache for faster JIT in frozen app
try:
    import numba
    numba_pkg = os.path.dirname(numba.__file__)
    added_files.append((numba_pkg, 'numba'))
except ImportError:
    pass

# Hidden imports for torch and numba
hiddenimports = [
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'numpy',
    'scipy',
    'scipy.signal',
    'scipy.special',
    'soundfile',
    'noisereduce',
    'pedalboard',
    'customtkinter',
    'CTkMessagebox',
    'numba',
    'numba.core',
    'tkinter',
    'tkinter.filedialog',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
]

a = Analysis(
    ['gui_enhancer_v4.py'],
    pathex=[SPEC_DIR],
    binaries=[
        # Bundle FFmpeg
        (r'C:\cmd\ffmpeg\bin\ffmpeg.exe', 'ffmpeg'),
        (r'C:\cmd\ffmpeg\bin\ffprobe.exe', 'ffmpeg'),
    ],
    datas=added_files,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'torchaudio',  # Not used, causes DLL issues
        'torchvision', # Not used
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out torchaudio binaries and datas
a.binaries = [x for x in a.binaries if 'torchaudio' not in x[0].lower()]
a.datas = [x for x in a.datas if 'torchaudio' not in x[0].lower()]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AudioStudioAIPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AudioStudioAIPro',
)
