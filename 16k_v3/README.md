# Audio Enhancement V12/V14/V18

## Quick Start

```bash
# Fast mode (V18) - recommended for batch
python enhance.py input.mp3 --fast

# Auto-select best V12/V14 mode
python enhance.py input.mp3 --auto

# Manual mode selection
python enhance.py input.mp3            # V12 (default)
python enhance.py input.mp3 --hf       # V14 (+ HF reduction)
python enhance.py input.mp3 --2pass    # 2-pass (high noise)
python enhance.py input.mp3 --2pass --hf  # 2-pass + HF
```

## Mode Comparison

| Mode | Flag | Speed | Quality | Use Case |
|------|------|-------|---------|----------|
| V12 | (default) | ~15x | Good | Clean source |
| V14 | --hf | ~15x | Better | HF noise |
| 2-pass | --2pass | ~10x | Best | Noisy |
| **V18** | **--fast** | **28-48x** | Good | **Batch/Fast** |

## V18 Smart Mode

V18 automatically analyzes source and selects 1 or 2 V7 passes:
- Clean source (< -35 dB): 1 pass → 48x speed
- Noisy source (> -35 dB): 2 passes → 28x speed

## Files

- `enhance.py` - Main CLI with all modes
- `audio_processor_v12_blended.py` - V12 processor
- `audio_processor_v14_combined.py` - V14 processor
- `audio_processor_v18_smart.py` - V18 Fast processor
- `audio_processor_crnn_attention_latent.py` - V7 processor
- `core/crnn_attention.py` - CRNN model
- `core/crnn_attention.pth` - Model weights
