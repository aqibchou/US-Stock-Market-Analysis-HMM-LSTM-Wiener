"""
XGB-HMM-Wiener Model Module
Enhanced XGB-HMM with integrated Wiener filtering
"""

from .XGB_HMM_Wiener import (
    XGB_HMM_Wiener,
    train_xgb_hmm_wiener
)

__all__ = [
    'XGB_HMM_Wiener',
    'train_xgb_hmm_wiener'
]
