"""Gradient compression for bandwidth optimization."""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass
class CompressionStats:
    """Statistics for gradient compression."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    nnz: int  # Number of non-zeros


class TopKCompressor:
    """Top-K sparsification with error feedback.
    
    Keeps only the top K% of gradients by magnitude.
    Uses error feedback to accumulate residuals.
    """
    
    def __init__(self, ratio: float = 0.01):
        """
        Args:
            ratio: Fraction of gradients to keep (default 1%)
        """
        self.ratio = ratio
        self.error_feedback: Optional[np.ndarray] = None
    
    def compress(
        self, 
        gradients: np.ndarray,
        use_error_feedback: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, CompressionStats]:
        """Compress gradients using Top-K sparsification.
        
        Args:
            gradients: Dense gradient array
            use_error_feedback: Whether to use error feedback
        
        Returns:
            (values, indices, stats)
        """
        # Apply error feedback
        if use_error_feedback and self.error_feedback is not None:
            gradients = gradients + self.error_feedback
        
        # Compute K
        k = max(1, int(len(gradients) * self.ratio))
        
        # Get top-K indices by magnitude
        abs_grads = np.abs(gradients)
        indices = np.argpartition(abs_grads, -k)[-k:]
        indices = indices[np.argsort(abs_grads[indices])[::-1]]
        
        values = gradients[indices].astype(np.float32)
        
        # Store error feedback (residual)
        if use_error_feedback:
            self.error_feedback = gradients.copy()
            self.error_feedback[indices] = 0
        
        stats = CompressionStats(
            original_size=gradients.nbytes,
            compressed_size=values.nbytes + indices.nbytes,
            compression_ratio=len(values) / len(gradients),
            nnz=len(values),
        )
        
        return values, indices.astype(np.int64), stats
    
    def decompress(
        self,
        values: np.ndarray,
        indices: np.ndarray,
        size: int,
    ) -> np.ndarray:
        """Decompress sparse gradients to dense.
        
        Args:
            values: Non-zero values
            indices: Indices of non-zero values
            size: Original gradient size
        
        Returns:
            Dense gradient array
        """
        gradients = np.zeros(size, dtype=np.float32)
        gradients[indices] = values
        return gradients
    
    def reset_error_feedback(self) -> None:
        """Reset accumulated error feedback."""
        self.error_feedback = None


class RandomKCompressor:
    """Random sparsification.
    
    Randomly selects K% of gradients, scaled to maintain expectation.
    """
    
    def __init__(self, ratio: float = 0.01):
        self.ratio = ratio
    
    def compress(
        self, 
        gradients: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, CompressionStats]:
        """Compress using random selection."""
        k = max(1, int(len(gradients) * self.ratio))
        
        # Random selection
        indices = np.random.choice(len(gradients), k, replace=False)
        values = gradients[indices] / self.ratio  # Scale to maintain expectation
        
        stats = CompressionStats(
            original_size=gradients.nbytes,
            compressed_size=values.nbytes + indices.nbytes,
            compression_ratio=len(values) / len(gradients),
            nnz=len(values),
        )
        
        return values.astype(np.float32), indices.astype(np.int64), stats
    
    def decompress(
        self,
        values: np.ndarray,
        indices: np.ndarray,
        size: int,
    ) -> np.ndarray:
        """Decompress sparse gradients."""
        gradients = np.zeros(size, dtype=np.float32)
        gradients[indices] = values * self.ratio  # Unscale
        return gradients


class QuantizationCompressor:
    """Gradient quantization to lower precision."""
    
    def __init__(self, bits: int = 8):
        """
        Args:
            bits: Number of bits for quantization (1-8)
        """
        self.bits = bits
        self.levels = 2 ** bits
    
    def compress(
        self, 
        gradients: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[float, float], CompressionStats]:
        """Quantize gradients.
        
        Returns:
            (quantized_values, (min_val, max_val), stats)
        """
        min_val = gradients.min()
        max_val = gradients.max()
        
        # Normalize to [0, 1]
        if max_val - min_val > 0:
            normalized = (gradients - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(gradients)
        
        # Quantize
        quantized = np.round(normalized * (self.levels - 1)).astype(np.uint8)
        
        stats = CompressionStats(
            original_size=gradients.nbytes,
            compressed_size=quantized.nbytes + 16,  # +16 for min/max
            compression_ratio=quantized.nbytes / gradients.nbytes,
            nnz=len(gradients),
        )
        
        return quantized, (float(min_val), float(max_val)), stats
    
    def decompress(
        self,
        quantized: np.ndarray,
        params: Tuple[float, float],
    ) -> np.ndarray:
        """Dequantize gradients."""
        min_val, max_val = params
        
        # Dequantize
        normalized = quantized.astype(np.float32) / (self.levels - 1)
        gradients = normalized * (max_val - min_val) + min_val
        
        return gradients


def compress_gradients(
    gradients: np.ndarray,
    method: str = "topk",
    ratio: float = 0.01,
    bits: int = 8,
) -> Tuple[np.ndarray, np.ndarray, CompressionStats]:
    """Convenience function for gradient compression.
    
    Args:
        gradients: Dense gradient array
        method: Compression method ("topk", "random", "quantize")
        ratio: Sparsification ratio (for topk/random)
        bits: Quantization bits
    
    Returns:
        (values, indices_or_params, stats)
    """
    if method == "topk":
        compressor = TopKCompressor(ratio)
        return compressor.compress(gradients)
    elif method == "random":
        compressor = RandomKCompressor(ratio)
        return compressor.compress(gradients)
    elif method == "quantize":
        compressor = QuantizationCompressor(bits)
        return compressor.compress(gradients)
    else:
        raise ValueError(f"Unknown compression method: {method}")
