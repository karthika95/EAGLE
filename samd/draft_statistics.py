"""
Draft Token Statistics Tracking
Tracks detailed statistics for draft token generation and acceptance across
different methods (static SAM, dynamic SAM, EAGLE/tree).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import numpy as np


@dataclass
class StepStatistics:
    """Statistics for a single generation step"""
    step: int
    draft_method: str  # "static", "dynamic", "tree"
    draft_tokens_count: int  # Number of draft tokens generated
    accepted_tokens_count: int  # Number of tokens accepted (< draft_tokens_count)
    acceptance_rate: float = 0.0  # accepted / drafted
    draft_indices: List[int] = field(default_factory=list)  # Token indices drafted
    accepted_indices: List[int] = field(default_factory=list)  # Token indices accepted
    tree_depth: Optional[int] = None  # For tree drafting only
    static_match_length: Optional[int] = None  # For static SAM
    dynamic_match_length: Optional[int] = None  # For dynamic SAM
    
    def __post_init__(self):
        if self.draft_tokens_count > 0:
            self.acceptance_rate = self.accepted_tokens_count / self.draft_tokens_count

    def compute_metrics(self):
        """Compute derived metrics for this step."""
        if self.draft_tokens_count > 0:
            self.acceptance_rate = self.accepted_tokens_count / self.draft_tokens_count
        else:
            self.acceptance_rate = 0.0


@dataclass
class SequenceStatistics:
    """Statistics for a complete sequence generation (one question/answer)"""
    sequence_id: int
    total_steps: int
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_output_tokens: int = 0
    
    # Per-method statistics
    static_draft_count: int = 0
    static_accepted_count: int = 0
    dynamic_draft_count: int = 0
    dynamic_accepted_count: int = 0
    tree_draft_count: int = 0
    tree_accepted_count: int = 0
    
    # Per-method step counts
    static_steps: int = 0
    dynamic_steps: int = 0
    tree_steps: int = 0
    
    # Acceptance rates
    static_acceptance_rate: float = 0.0
    dynamic_acceptance_rate: float = 0.0
    tree_acceptance_rate: float = 0.0
    overall_acceptance_rate: float = 0.0
    
    # Efficiency metrics
    speedup: float = 1.0  # vs baseline (decode_steps / total_steps)
    tokens_per_step: float = 0.0  # total_output_tokens / total_steps
    
    step_stats: List[StepStatistics] = field(default_factory=list)
    
    def compute_metrics(self):
        """Compute derived metrics"""
        if self.total_steps > 0:
            self.tokens_per_step = self.total_output_tokens / self.total_steps
        
        if self.static_draft_count > 0:
            self.static_acceptance_rate = self.static_accepted_count / self.static_draft_count
        
        if self.dynamic_draft_count > 0:
            self.dynamic_acceptance_rate = self.dynamic_accepted_count / self.dynamic_draft_count
        
        if self.tree_draft_count > 0:
            self.tree_acceptance_rate = self.tree_accepted_count / self.tree_draft_count
        
        if self.total_draft_tokens > 0:
            self.overall_acceptance_rate = self.total_accepted_tokens / self.total_draft_tokens
        
        # Speedup proxy vs baseline autoregressive decoding (1 token/step).
        if self.total_steps > 0:
            self.speedup = self.total_output_tokens / self.total_steps
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "sequence_id": self.sequence_id,
            "total_steps": self.total_steps,
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "total_output_tokens": self.total_output_tokens,
            "static": {
                "draft_count": self.static_draft_count,
                "accepted_count": self.static_accepted_count,
                "steps": self.static_steps,
                "acceptance_rate": self.static_acceptance_rate,
            },
            "dynamic": {
                "draft_count": self.dynamic_draft_count,
                "accepted_count": self.dynamic_accepted_count,
                "steps": self.dynamic_steps,
                "acceptance_rate": self.dynamic_acceptance_rate,
            },
            "tree": {
                "draft_count": self.tree_draft_count,
                "accepted_count": self.tree_accepted_count,
                "steps": self.tree_steps,
                "acceptance_rate": self.tree_acceptance_rate,
            },
            "overall_acceptance_rate": self.overall_acceptance_rate,
            "speedup": self.speedup,
            "tokens_per_step": self.tokens_per_step,
        }


@dataclass
class BatchStatistics:
    """Statistics for a batch of sequences"""
    model_id: str
    dataset_name: str
    total_sequences: int = 0
    
    # Aggregate statistics
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_output_tokens: int = 0
    total_steps: int = 0
    
    # Per-method aggregates
    static_draft_total: int = 0
    static_accepted_total: int = 0
    static_steps_total: int = 0
    
    dynamic_draft_total: int = 0
    dynamic_accepted_total: int = 0
    dynamic_steps_total: int = 0
    
    tree_draft_total: int = 0
    tree_accepted_total: int = 0
    tree_steps_total: int = 0
    
    # Sequence-level statistics
    sequence_stats: List[SequenceStatistics] = field(default_factory=list)
    
    # Aggregated metrics
    static_acceptance_rate: float = 0.0
    dynamic_acceptance_rate: float = 0.0
    tree_acceptance_rate: float = 0.0
    overall_acceptance_rate: float = 0.0
    
    speedup_mean: float = 0.0
    speedup_std: float = 0.0
    tokens_per_step_mean: float = 0.0
    tokens_per_step_std: float = 0.0
    
    def add_sequence(self, seq_stats: SequenceStatistics):
        """Add a sequence's statistics"""
        seq_stats.compute_metrics()
        self.sequence_stats.append(seq_stats)
        
        self.total_sequences += 1
        self.total_draft_tokens += seq_stats.total_draft_tokens
        self.total_accepted_tokens += seq_stats.total_accepted_tokens
        self.total_output_tokens += seq_stats.total_output_tokens
        self.total_steps += seq_stats.total_steps
        
        self.static_draft_total += seq_stats.static_draft_count
        self.static_accepted_total += seq_stats.static_accepted_count
        self.static_steps_total += seq_stats.static_steps
        
        self.dynamic_draft_total += seq_stats.dynamic_draft_count
        self.dynamic_accepted_total += seq_stats.dynamic_accepted_count
        self.dynamic_steps_total += seq_stats.dynamic_steps
        
        self.tree_draft_total += seq_stats.tree_draft_count
        self.tree_accepted_total += seq_stats.tree_accepted_count
        self.tree_steps_total += seq_stats.tree_steps
    
    def compute_metrics(self):
        """Compute aggregate metrics"""
        if self.static_draft_total > 0:
            self.static_acceptance_rate = self.static_accepted_total / self.static_draft_total
        
        if self.dynamic_draft_total > 0:
            self.dynamic_acceptance_rate = self.dynamic_accepted_total / self.dynamic_draft_total
        
        if self.tree_draft_total > 0:
            self.tree_acceptance_rate = self.tree_accepted_total / self.tree_draft_total
        
        if self.total_draft_tokens > 0:
            self.overall_acceptance_rate = self.total_accepted_tokens / self.total_draft_tokens
        
        # Compute mean and std of speedup and tokens per step
        if self.sequence_stats:
            speedups = [s.speedup for s in self.sequence_stats]
            tokens_per_steps = [s.tokens_per_step for s in self.sequence_stats]
            
            self.speedup_mean = np.mean(speedups)
            self.speedup_std = np.std(speedups)
            self.tokens_per_step_mean = np.mean(tokens_per_steps)
            self.tokens_per_step_std = np.std(tokens_per_steps)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        self.compute_metrics()
        return {
            "model_id": self.model_id,
            "dataset_name": self.dataset_name,
            "total_sequences": self.total_sequences,
            "total_steps": self.total_steps,
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "total_output_tokens": self.total_output_tokens,
            "static": {
                "draft_total": self.static_draft_total,
                "accepted_total": self.static_accepted_total,
                "steps_total": self.static_steps_total,
                "acceptance_rate": self.static_acceptance_rate,
            },
            "dynamic": {
                "draft_total": self.dynamic_draft_total,
                "accepted_total": self.dynamic_accepted_total,
                "steps_total": self.dynamic_steps_total,
                "acceptance_rate": self.dynamic_acceptance_rate,
            },
            "tree": {
                "draft_total": self.tree_draft_total,
                "accepted_total": self.tree_accepted_total,
                "steps_total": self.tree_steps_total,
                "acceptance_rate": self.tree_acceptance_rate,
            },
            "overall_acceptance_rate": self.overall_acceptance_rate,
            "speedup": {
                "mean": self.speedup_mean,
                "std": self.speedup_std,
            },
            "tokens_per_step": {
                "mean": self.tokens_per_step_mean,
                "std": self.tokens_per_step_std,
            },
        }
    
    def print_report(self):
        """Print a detailed report"""
        self.compute_metrics()
        
        print("\n" + "="*80)
        print(f"Draft Token Statistics Report: {self.model_id} on {self.dataset_name}")
        print("="*80)
        
        print(f"\nTotal Sequences: {self.total_sequences}")
        print(f"Total Generation Steps: {self.total_steps}")
        print(f"Total Tokens Generated: {self.total_output_tokens}")
        
        print(f"\n{'Method':<15} {'Draft':<12} {'Accepted':<12} {'Steps':<10} {'Acc. Rate':<12}")
        print("-" * 65)
        
        print(f"{'Static SAM':<15} {self.static_draft_total:<12} {self.static_accepted_total:<12} {self.static_steps_total:<10} {self.static_acceptance_rate:<12.2%}")
        print(f"{'Dynamic SAM':<15} {self.dynamic_draft_total:<12} {self.dynamic_accepted_total:<12} {self.dynamic_steps_total:<10} {self.dynamic_acceptance_rate:<12.2%}")
        print(f"{'Tree/EAGLE':<15} {self.tree_draft_total:<12} {self.tree_accepted_total:<12} {self.tree_steps_total:<10} {self.tree_acceptance_rate:<12.2%}")
        print("-" * 65)
        print(f"{'TOTAL':<15} {self.total_draft_tokens:<12} {self.total_accepted_tokens:<12} {self.total_steps:<10} {self.overall_acceptance_rate:<12.2%}")
        
        print(f"\nEfficiency Metrics:")
        print(f"  Average Speedup: {self.speedup_mean:.4f}x (±{self.speedup_std:.4f})")
        print(f"  Tokens/Step: {self.tokens_per_step_mean:.4f} (±{self.tokens_per_step_std:.4f})")
        
        print("\n" + "="*80)
