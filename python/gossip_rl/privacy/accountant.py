"""Privacy Accountants for tracking privacy budget consumption."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import special


@dataclass
class PrivacySpent:
    """Record of privacy spent in a single step."""
    epsilon: float
    delta: float
    noise_multiplier: float
    sample_rate: float
    step: int


class PrivacyAccountant(ABC):
    """Abstract base class for privacy accountants."""
    
    def __init__(self):
        self.history: List[PrivacySpent] = []
        self._step_count = 0
    
    @abstractmethod
    def step(
        self,
        noise_multiplier: float,
        sample_rate: float = 1.0,
    ) -> None:
        """Record a single training step's privacy cost."""
        pass
    
    @abstractmethod
    def get_epsilon(self, delta: float) -> float:
        """Get current ε given target δ."""
        pass
    
    def get_privacy_spent(self, delta: float) -> Tuple[float, float]:
        """Get (ε, δ) privacy spent."""
        return self.get_epsilon(delta), delta
    
    @property
    def step_count(self) -> int:
        """Get number of steps recorded."""
        return self._step_count
    
    def is_budget_exhausted(self, max_epsilon: float, delta: float) -> bool:
        """Check if privacy budget is exhausted."""
        return self.get_epsilon(delta) > max_epsilon


class RDPAccountant(PrivacyAccountant):
    """Rényi Differential Privacy (RDP) Accountant.
    
    Uses RDP for tight composition, then converts to (ε, δ)-DP.
    
    RDP provides tighter bounds than naive composition:
    - RDP of order α: D_α(M(x) || M(x')) ≤ ε_α
    - Composition: ε_α(M₁ ∘ M₂) ≤ ε_α(M₁) + ε_α(M₂)
    - Convert to (ε,δ)-DP: ε = min_α [ε_α - log(δ)/(α-1)]
    """
    
    def __init__(self, orders: Optional[List[float]] = None):
        super().__init__()
        
        if orders is None:
            # Default orders covering common ranges
            self.orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        else:
            self.orders = orders
        
        # Running RDP epsilon for each order
        self.rdp_epsilon = np.zeros(len(self.orders))
    
    def step(
        self,
        noise_multiplier: float,
        sample_rate: float = 1.0,
    ) -> None:
        """Record a single step and accumulate RDP."""
        self._step_count += 1
        
        # Compute RDP for this step
        rdp = self._compute_rdp(noise_multiplier, sample_rate)
        self.rdp_epsilon += rdp
        
        self.history.append(PrivacySpent(
            epsilon=0,  # Will be computed on demand
            delta=0,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            step=self._step_count,
        ))
    
    def _compute_rdp(
        self,
        noise_multiplier: float,
        sample_rate: float,
    ) -> np.ndarray:
        """Compute RDP for Gaussian mechanism with subsampling.
        
        Uses the analytical moment accountant formula for subsampled Gaussian.
        """
        rdp = np.zeros(len(self.orders))
        
        for i, order in enumerate(self.orders):
            if sample_rate == 1.0:
                # No subsampling: standard Gaussian RDP
                rdp[i] = order / (2 * noise_multiplier ** 2)
            else:
                # Subsampled Gaussian: use analytical formula
                rdp[i] = self._compute_subsampled_rdp(
                    noise_multiplier, sample_rate, order
                )
        
        return rdp
    
    def _compute_subsampled_rdp(
        self,
        sigma: float,
        q: float,
        alpha: float,
    ) -> float:
        """Compute RDP of subsampled Gaussian mechanism.
        
        Using Proposition 3 from "Rényi Differential Privacy of the
        Sampled Gaussian Mechanism" (Mironov, 2017).
        """
        if alpha <= 1:
            return 0.0
        
        if q == 0:
            return 0.0
        
        if q == 1:
            return alpha / (2 * sigma ** 2)
        
        # For small q, use log-space computation for numerical stability
        # RDP ≈ q² * α / (2σ²) for small q
        
        # General formula using moment generating function
        log_terms = []
        for k in range(min(int(alpha) + 1, 100)):
            log_comb = self._log_comb(alpha, k)
            log_term = (
                log_comb
                + k * math.log(q)
                + (alpha - k) * math.log(1 - q)
                + k * (k - 1) / (2 * sigma ** 2)
            )
            log_terms.append(log_term)
        
        # Log-sum-exp for numerical stability
        max_log = max(log_terms)
        result = max_log + math.log(sum(math.exp(lt - max_log) for lt in log_terms))
        
        return result / (alpha - 1)
    
    def _log_comb(self, n: float, k: int) -> float:
        """Compute log of binomial coefficient."""
        if k == 0:
            return 0.0
        if k > n:
            return float('-inf')
        
        # Use log-gamma for non-integer n
        return (
            special.gammaln(n + 1)
            - special.gammaln(k + 1)
            - special.gammaln(n - k + 1)
        )
    
    def get_epsilon(self, delta: float) -> float:
        """Convert RDP to (ε, δ)-DP.
        
        ε = min_α [ε_α(α) + log(1/δ)/(α-1) - log(α)/(α-1)]
        """
        if delta <= 0:
            return float('inf')
        
        log_delta = math.log(delta)
        
        eps_list = []
        for i, alpha in enumerate(self.orders):
            if alpha <= 1:
                continue
            
            eps = self.rdp_epsilon[i] + (log_delta + math.log(alpha)) / (1 - alpha)
            # Alternative formula: rdp - log(delta * (alpha - 1)) / alpha
            
            if eps > 0:
                eps_list.append(eps)
        
        if not eps_list:
            return float('inf')
        
        return min(eps_list)
    
    def get_optimal_order(self, delta: float) -> float:
        """Get the optimal RDP order for current privacy spent."""
        log_delta = math.log(delta)
        
        min_eps = float('inf')
        best_order = self.orders[0]
        
        for i, alpha in enumerate(self.orders):
            if alpha <= 1:
                continue
            
            eps = self.rdp_epsilon[i] + log_delta / (alpha - 1)
            if eps < min_eps:
                min_eps = eps
                best_order = alpha
        
        return best_order


class GLWAccountant(PrivacyAccountant):
    """Privacy Loss Distribution (PLD) based accountant.
    
    Uses the Gaussian-based privacy loss distribution for
    tighter accounting with numerical integration.
    """
    
    def __init__(
        self,
        discretization: float = 1e-4,
        truncation: float = 500.0,
    ):
        super().__init__()
        self.discretization = discretization
        self.truncation = truncation
        
        # Privacy loss distribution (discretized)
        self.pld: Optional[np.ndarray] = None
        self.pld_domain: Optional[np.ndarray] = None
    
    def step(
        self,
        noise_multiplier: float,
        sample_rate: float = 1.0,
    ) -> None:
        """Record step by convolving PLDs."""
        self._step_count += 1
        
        # Compute PLD for this step
        step_pld = self._compute_gaussian_pld(noise_multiplier, sample_rate)
        
        if self.pld is None:
            self.pld = step_pld
        else:
            # Convolve PLDs
            self.pld = np.convolve(self.pld, step_pld, mode='full')
            # Renormalize
            self.pld = self.pld / self.pld.sum()
        
        self.history.append(PrivacySpent(
            epsilon=0,
            delta=0,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            step=self._step_count,
        ))
    
    def _compute_gaussian_pld(
        self,
        sigma: float,
        q: float,
    ) -> np.ndarray:
        """Compute discretized PLD for Gaussian mechanism."""
        # Domain of privacy loss
        n_points = int(2 * self.truncation / self.discretization)
        domain = np.linspace(-self.truncation, self.truncation, n_points)
        
        self.pld_domain = domain
        
        # For Gaussian mechanism, privacy loss is O(1/σ²)
        # PDF of privacy loss random variable
        # Using numerical approximation
        
        mu = 1 / (2 * sigma ** 2)  # Expected privacy loss
        std = 1 / sigma  # Std of privacy loss
        
        # Approximate as Gaussian (valid for large σ)
        pld = np.exp(-0.5 * ((domain - mu) / std) ** 2)
        pld = pld / pld.sum()
        
        return pld
    
    def get_epsilon(self, delta: float) -> float:
        """Get epsilon from PLD."""
        if self.pld is None or self.pld_domain is None:
            return 0.0
        
        # Find minimum epsilon such that P[L > ε] ≤ δ
        cdf = np.cumsum(self.pld)
        
        for i, eps in enumerate(self.pld_domain):
            if 1 - cdf[i] <= delta:
                return max(0.0, eps)
        
        return self.pld_domain[-1]
