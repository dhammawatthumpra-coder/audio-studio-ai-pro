"""
Frequency Band OM-LSA for 16kHz
================================
Band-specific processing for native 16kHz.
"""

import numpy as np


# Band configuration for 16kHz (Nyquist = 8kHz)
# More aggressive on high bands - 4-8kHz is where crickets/cicadas live
BAND_CONFIGS_16K = {
    "low": {
        "range": (0, 500),
        "oversubtract": 2.5,
        "floor": 0.08,
    },
    "mid_low": {
        "range": (500, 1500),
        "oversubtract": 1.6,
        "floor": 0.15,
    },
    "mid": {
        "range": (1500, 3500),
        "oversubtract": 1.0,  # Protect speech
        "floor": 0.18,
    },
    "mid_high": {
        "range": (3500, 5500),
        "oversubtract": 4.0,  # More aggressive
        "floor": 0.03,
    },
    "high": {
        "range": (5500, 8000),
        "oversubtract": 5.0,  # Even more aggressive
        "floor": 0.01,
    },
}


class FrequencyBandOMLSA16k:
    """Frequency-band specific processing for 16kHz."""
    
    def __init__(self, n_bins: int = 257, sample_rate: int = 16000):
        self.n_bins = n_bins
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        
        # Create band masks
        self.band_masks = {}
        self.band_oversub = {}
        self.band_floors = {}
        
        for band_name, config in BAND_CONFIGS_16K.items():
            low_hz, high_hz = config["range"]
            low_bin = int(low_hz / self.nyquist * n_bins)
            high_bin = int(high_hz / self.nyquist * n_bins)
            
            mask = np.zeros(n_bins, dtype=bool)
            mask[low_bin:high_bin] = True
            
            self.band_masks[band_name] = mask
            self.band_oversub[band_name] = config["oversubtract"]
            self.band_floors[band_name] = config["floor"]
    
    def get_band_oversubtract(self, spp: float) -> np.ndarray:
        """Get per-bin oversubtraction factors."""
        oversub = np.ones(self.n_bins)
        
        # SPP-dependent suppression:
        # Only protect during clear speech, otherwise suppress
        speech_protection = 1.0 - (spp * 0.4)
        
        for band_name, mask in self.band_masks.items():
            base = self.band_oversub[band_name]
            
            if band_name in ["mid_high", "high"]:
                # High bands: only protect during CLEAR speech
                if spp > 0.7:  # Even higher threshold
                    oversub[mask] = base * 0.6  # Less protection
                else:
                    oversub[mask] = base  # Full suppression
            else:
                # Lower bands: normal behavior
                oversub[mask] = base * speech_protection
        
        return oversub
    
    def apply_gain(self, gain: np.ndarray, spp: float) -> np.ndarray:
        """Apply band-specific gain floors."""
        result = gain.copy()
        
        for band_name, mask in self.band_masks.items():
            floor = self.band_floors[band_name]
            # Higher floor during speech for speech-relevant frequencies
            # Extended to include mid_high (3.5-5.5kHz) for better voice harmonics
            if band_name in ["mid_low", "mid", "mid_high"]:
                floor = floor + (spp * 0.08)
            result[mask] = np.maximum(result[mask], floor)
        
        return result
