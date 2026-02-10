"""
Reward Hacking Detection and Prevention Module

This module adds to notebook 03_reward_design_fundamentals.ipynb
with implementations for detecting and preventing reward hacking.

Run in notebook with:
    %run reward_hacking_detection.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict


# =============================================================================
# SECTION 7: REWARD HACKING DETECTION AND PREVENTION
# =============================================================================

# Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure."
#
# | Hacking Type       | Symptom                  | Detection                    |
# |--------------------|--------------------------|------------------------------|
# | Length hacking     | Excessive verbosity      | Length vs. accuracy diverge  |
# | Sycophancy         | Always agrees            | Human eval vs. proxy mismatch|
# | Format mimicry     | Copies style, not content| Gold RM score drops          |
# | Overoptimization   | Proxy ↑ but gold ↓       | Training curve divergence    |


class RewardHackingDetector:
    """
    Detect reward hacking during training.
    
    Key signals:
    - Proxy reward increasing while gold reward plateaus/drops
    - KL divergence exploding
    - Response length inflating
    - Diversity collapsing
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.proxy_rewards = []
        self.gold_rewards = []  # From held-out evaluation
        self.kl_values = []
        self.lengths = []
    
    def update(self, proxy: float, gold: float, kl: float, length: int):
        """Record metrics for one training step."""
        self.proxy_rewards.append(proxy)
        self.gold_rewards.append(gold)
        self.kl_values.append(kl)
        self.lengths.append(length)
    
    def detect_overoptimization(self) -> Dict:
        """
        Detect if proxy is being overoptimized.
        
        Returns:
            Dict with detection results and metrics
        """
        if len(self.proxy_rewards) < self.window_size * 2:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        # Compare recent window to earlier window
        early_proxy = np.mean(self.proxy_rewards[:self.window_size])
        late_proxy = np.mean(self.proxy_rewards[-self.window_size:])
        
        early_gold = np.mean(self.gold_rewards[:self.window_size])
        late_gold = np.mean(self.gold_rewards[-self.window_size:])
        
        proxy_delta = late_proxy - early_proxy
        gold_delta = late_gold - early_gold
        
        # Hacking detected if proxy goes up but gold goes down
        detected = proxy_delta > 0.1 and gold_delta < -0.05
        
        return {
            'detected': detected,
            'proxy_delta': proxy_delta,
            'gold_delta': gold_delta,
            'reason': 'Proxy ↑ while gold ↓' if detected else 'OK'
        }
    
    def detect_length_hacking(self, baseline_length: float = 200) -> Dict:
        """Detect if model is gaming length."""
        if len(self.lengths) < self.window_size:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        avg_length = np.mean(self.lengths[-self.window_size:])
        ratio = avg_length / baseline_length
        
        detected = ratio > 2.0  # More than 2x baseline
        
        return {
            'detected': detected,
            'avg_length': avg_length,
            'length_ratio': ratio,
            'reason': f'Length {ratio:.1f}x baseline' if detected else 'OK'
        }
    
    def detect_kl_explosion(self, threshold: float = 15.0) -> Dict:
        """Detect if KL divergence is exploding."""
        if len(self.kl_values) < 10:
            return {'detected': False, 'reason': 'Insufficient data'}
        
        recent_kl = np.mean(self.kl_values[-10:])
        detected = recent_kl > threshold
        
        return {
            'detected': detected,
            'recent_kl': recent_kl,
            'reason': f'KL={recent_kl:.1f} > {threshold}' if detected else 'OK'
        }
    
    def should_early_stop(self) -> Tuple[bool, str]:
        """Determine if training should stop due to hacking."""
        overopt = self.detect_overoptimization()
        length = self.detect_length_hacking()
        kl = self.detect_kl_explosion()
        
        if overopt['detected']:
            return True, f"Overoptimization: {overopt['reason']}"
        if kl['detected']:
            return True, f"KL explosion: {kl['reason']}"
        if length['detected']:
            return True, f"Length hacking: {length['reason']}"
        
        return False, "Training healthy"


def simulate_goodharts_law(steps: int = 200):
    """
    Simulate what happens when proxy reward is overoptimized.
    
    Early training: Both proxy and gold improve together
    Mid training: Gold plateaus
    Late training: Proxy keeps rising, gold drops (HACKING!)
    """
    np.random.seed(42)
    detector = RewardHackingDetector(window_size=30)
    
    print("Simulating Goodhart's Law (Overoptimization):")
    print(f"{'Step':>6} {'Proxy':>8} {'Gold':>8} {'Status':>20}")
    print("-" * 45)
    
    for step in range(steps):
        # Simulate training dynamics
        if step < 50:  # Early: Both improve
            proxy = 0.3 + step * 0.01 + np.random.normal(0, 0.02)
            gold = 0.3 + step * 0.008 + np.random.normal(0, 0.02)
        elif step < 100:  # Mid: Gold plateaus
            proxy = 0.8 + (step - 50) * 0.005 + np.random.normal(0, 0.02)
            gold = 0.7 + np.random.normal(0, 0.02)  # Plateau!
        else:  # Late: Gold drops (HACKING)
            proxy = 1.0 + (step - 100) * 0.003 + np.random.normal(0, 0.02)
            gold = 0.7 - (step - 100) * 0.003 + np.random.normal(0, 0.02)  # Drops!
        
        kl = 3.0 + step * 0.05 + np.random.normal(0, 0.5)
        length = int(200 + step * 1.5)
        
        detector.update(proxy, gold, kl, length)
        
        if step % 40 == 0 or step == steps - 1:
            should_stop, reason = detector.should_early_stop()
            status = f"⚠️ {reason}" if should_stop else "✓ OK"
            print(f"{step:>6} {proxy:>8.3f} {gold:>8.3f} {status:>20}")
    
    print("\n" + "=" * 45)
    final_check = detector.detect_overoptimization()
    print(f"Final verdict: {'🚨 HACKING DETECTED' if final_check['detected'] else '✓ OK'}")
    print(f"  Proxy Δ: {final_check.get('proxy_delta', 0):.3f}")
    print(f"  Gold Δ: {final_check.get('gold_delta', 0):.3f}")


# =============================================================================
# 7.2 ENSEMBLE REWARD MODEL (HACK RESISTANCE)
# =============================================================================

class EnsembleRewardModel:
    """
    Combine multiple reward models to resist hacking.
    
    Intuition: Harder to fool multiple models simultaneously.
    Each RM has different biases → average cancels them out.
    """
    
    def __init__(self, reward_models: List[nn.Module]):
        self.models = reward_models
        self.n_models = len(reward_models)
    
    def compute_reward(self, input_ids: torch.Tensor) -> Dict:
        """
        Compute ensemble reward.
        
        Returns:
            Dict with 'mean', 'std', 'individual' rewards
        """
        rewards = []
        
        with torch.no_grad():
            for model in self.models:
                r = model(input_ids)
                rewards.append(r)
        
        stacked = torch.stack(rewards, dim=0)  # (n_models, batch)
        
        return {
            'mean': stacked.mean(dim=0),  # Average across models
            'std': stacked.std(dim=0),    # Uncertainty
            'individual': rewards,
        }
    
    def compute_conservative_reward(self, input_ids: torch.Tensor,
                                    pessimism: float = 1.0) -> torch.Tensor:
        """
        Conservative reward: mean - pessimism * std
        
        Higher pessimism → more conservative → less hackable
        """
        result = self.compute_reward(input_ids)
        return result['mean'] - pessimism * result['std']


# =============================================================================
# 7.3 COMPLETE MITIGATION STRATEGY
# =============================================================================

class RewardHackingMitigation:
    """
    Complete anti-hacking system combining all strategies:
    
    1. KL penalty: Prevent drift from reference
    2. Length penalty: Prevent verbosity gaming
    3. Ensemble averaging: Reduce exploitability
    4. Early stopping: Detect and halt overoptimization
    """
    
    def __init__(self,
                 kl_beta: float = 0.1,
                 target_length: int = 200,
                 detector_window: int = 100):
        
        self.kl_beta = kl_beta
        self.target_length = target_length
        self.detector = RewardHackingDetector(window_size=detector_window)
    
    def compute_safe_reward(self,
                            raw_reward: float,
                            response_length: int,
                            kl_divergence: float,
                            gold_reward: Optional[float] = None) -> Dict:
        """
        Compute reward with all anti-hacking measures applied.
        """
        result = {'raw_reward': raw_reward}
        
        # 1. Length penalty (ratio method)
        if response_length <= self.target_length:
            length_penalized = raw_reward
        else:
            ratio = self.target_length / response_length
            length_penalized = raw_reward * ratio
        result['after_length_penalty'] = length_penalized
        
        # 2. KL penalty
        kl_penalty = self.kl_beta * kl_divergence
        kl_penalized = length_penalized - kl_penalty
        result['after_kl_penalty'] = kl_penalized
        
        # 3. Update detector
        # In production: gold_reward comes from held-out evaluation
        if gold_reward is None:
            gold_reward = raw_reward * 0.9  # Simulated
        self.detector.update(raw_reward, gold_reward, kl_divergence, response_length)
        
        # 4. Check for early stopping
        should_stop, reason = self.detector.should_early_stop()
        result['should_stop'] = should_stop
        result['stop_reason'] = reason
        
        result['final_reward'] = kl_penalized
        return result


def demo_mitigation_system():
    """Test the complete mitigation system."""
    
    mitigation = RewardHackingMitigation(
        kl_beta=0.1,
        target_length=200,
        detector_window=20,
    )
    
    print("Anti-Hacking Reward Pipeline:")
    print(f"{'Step':>5} {'Raw':>8} {'After Len':>10} {'After KL':>10} {'Status':>15}")
    print("-" * 55)
    
    for step in range(50):
        # Simulate increasingly hacky behavior
        raw_reward = 0.5 + step * 0.02
        length = 200 + step * 10  # Growing length
        kl = 3.0 + step * 0.2     # Growing KL
        
        result = mitigation.compute_safe_reward(
            raw_reward=raw_reward,
            response_length=length,
            kl_divergence=kl,
        )
        
        if step % 10 == 0 or result['should_stop']:
            status = '⚠️ STOP' if result['should_stop'] else '✓ OK'
            print(f"{step:>5} {raw_reward:>8.3f} {result['after_length_penalty']:>10.3f} "
                  f"{result['after_kl_penalty']:>10.3f} {status:>15}")
            if result['should_stop']:
                print(f"\n🚨 Early stop triggered: {result['stop_reason']}")
                break


# =============================================================================
# RUN DEMOS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SECTION 7: REWARD HACKING DETECTION AND PREVENTION")
    print("=" * 60)
    print()
    
    print("7.1 Goodhart's Law Simulation")
    print("-" * 40)
    simulate_goodharts_law()
    
    print("\n" + "=" * 60)
    print("7.3 Complete Mitigation System Demo")
    print("-" * 40)
    demo_mitigation_system()
    
    print("\n" + "=" * 60)
    print("SUMMARY: REWARD HACKING PREVENTION")
    print("=" * 60)
    print("""
Key Components Implemented:
1. RewardHackingDetector - Monitors for overoptimization
2. EnsembleRewardModel - Multiple RMs for hack resistance  
3. RewardHackingMitigation - Complete anti-hacking pipeline

Detection Signals:
- Proxy ↑ while Gold ↓ (Goodhart's Law)
- KL divergence explosion
- Response length inflation

Prevention Strategies:
- KL penalty: Prevent drift from reference model
- Length penalty: Penalize verbosity gaming
- Ensemble averaging: Reduce individual RM exploitability
- Early stopping: Halt when hacking detected
""")
