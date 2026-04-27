import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional, Tuple, Any, Set, Union
import logging
import copy
from pathlib import Path
from safetensors.torch import save_file, load_file
import json
from sklearn.cluster import KMeans
from huggingface_hub import ModelHubMixin, hf_hub_download
import os
import shutil

from .models import Example, AdaptiveHead, ModelConfig
from .memory import PrototypeMemory
from .ewc import EWC
from .strategic import (
    StrategicCostFunction, CostFunctionFactory, StrategicOptimizer, StrategicEvaluator
)


logger = logging.getLogger(__name__)

class AdaptiveClassifier(ModelHubMixin):
    """A flexible classifier that can adapt to new classes and examples."""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,  # Add seed parameter
        use_onnx: Optional[Union[bool, str]] = "auto",  # "auto", True, False
        trust_remote_code: bool = False
    ):
        """Initialize the adaptive classifier.

        Args:
            model_name: Name of the HuggingFace transformer model
            device: Device to run the model on (default: auto-detect)
            config: Optional configuration dictionary
            seed: Random seed for initialization
            use_onnx: Whether to use ONNX Runtime ("auto", True, False).
                     "auto" uses ONNX on CPU, PyTorch on GPU.
            trust_remote_code: Whether to trust remote code when loading models (default: False)
        """
        pass

    def _should_use_onnx(self, use_onnx: Union[bool, str]) -> bool:
        """Determine if ONNX should be used based on configuration and device.

        Args:
            use_onnx: User preference ("auto", True, False)

        Returns:
            True if ONNX should be used, False otherwise
        """
        pass

    def add_examples(self, texts: List[str], labels: List[str]):
        """Add new examples with special handling for new classes."""
        pass
    
    def _train_new_classes(self, old_head: Optional[nn.Module], new_classes: Set[str]):
        """Train the model with focus on new classes while preserving old class knowledge."""
        pass
    
    def _perform_strategic_training(self):
        """Perform strategic training on current examples."""
        pass
    
    def predict(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Predict with dual prediction system - blends strategic and regular predictions.
        
        If no cost function is provided, uses existing prediction logic (zero changes).
        If cost function is provided, blends strategic and regular predictions.
        
        Args:
            text: Input text to classify
            k: Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples
        """
        pass
    
    def _predict_regular(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Regular prediction logic (original implementation)."""
        pass
    
    def _predict_dual(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Dual prediction system that blends strategic and regular predictions."""
        pass
    
    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        include_onnx: bool = True,
        quantize_onnx: bool = True,
        **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Save the model to a directory.

        Args:
            save_directory: Directory to save the model to
            config: Optional additional configuration
            include_onnx: Whether to include ONNX export (default: True)
            quantize_onnx: Whether to quantize ONNX model (requires include_onnx=True)
            **kwargs: Additional arguments passed to save_pretrained

        Returns:
            Tuple of (dict of filenames, dict of objects to save)
        """
        pass

    @classmethod
    def _from_pretrained(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        use_onnx: Optional[Union[bool, str]] = "auto",
        prefer_quantized: bool = True,
        trust_remote_code: bool = False,
        **kwargs
    ) -> "AdaptiveClassifier":
        """Load a model from the HuggingFace Hub or local directory.

        Args:
            model_id: HuggingFace Hub model ID or path to local directory
            revision: Revision of the model on the Hub
            cache_dir: Cache directory for downloaded models
            force_download: Force download of models
            proxies: Proxies to use for downloading
            resume_download: Resume downloading if interrupted
            local_files_only: Use local files only, don't download
            token: Authentication token for Hub
            use_onnx: Whether to use ONNX Runtime ("auto", True, False)
            prefer_quantized: Use quantized ONNX model if available (default: True)
                             Set to False to use unquantized model for maximum accuracy
            trust_remote_code: Whether to trust remote code when loading models (default: False)
            **kwargs: Additional arguments passed to from_pretrained

        Returns:
            Loaded AdaptiveClassifier instance

        Examples:
            >>> # Load with quantized ONNX (default - faster, smaller)
            >>> classifier = AdaptiveClassifier.load("adaptive-classifier/llm-router")
            >>>
            >>> # Load with unquantized ONNX (maximum accuracy)
            >>> classifier = AdaptiveClassifier.load("adaptive-classifier/llm-router", prefer_quantized=False)
            >>>
            >>> # Force PyTorch (no ONNX)
            >>> classifier = AdaptiveClassifier.load("adaptive-classifier/llm-router", use_onnx=False)
            >>>
            >>> # Load model requiring custom code
            >>> classifier = AdaptiveClassifier.load("model-with-custom-code", trust_remote_code=True)
        """
        pass

    def _generate_model_card(self) -> str:
        """Generate a model card for the classifier.
        
        Returns:
            Model card content as string
        """
        pass

    def _format_class_distribution(self, stats: Dict[str, Any]) -> str:
        """Format class distribution for model card.
        
        Args:
            stats: Statistics from get_memory_stats()
            
        Returns:
            Formatted string of class distribution
        """
        pass

    def export_onnx(
        self,
        save_directory: Union[str, Path],
        quantize: bool = False,
        quantization_config: Optional[str] = "arm64"
    ) -> Path:
        """Export the transformer model to ONNX format.

        Args:
            save_directory: Directory to save ONNX model
            quantize: Whether to apply INT8 quantization
            quantization_config: Quantization configuration ("arm64", "avx512", "avx2")

        Returns:
            Path to the saved ONNX model directory

        Raises:
            ImportError: If optimum[onnxruntime] is not installed
            ValueError: If model is already in ONNX format
        """
        pass

    def push_to_hub(
        self,
        repo_id: str,
        include_onnx: bool = True,
        quantize_onnx: bool = True,
        token: Optional[str] = None,
        commit_message: Optional[str] = None,
        private: bool = False,
        **kwargs
    ):
        """Push model to HuggingFace Hub with ONNX export by default.

        Args:
            repo_id: Repository ID on HuggingFace Hub (e.g., "username/model-name")
            include_onnx: Whether to include ONNX version of the model (default: True)
            quantize_onnx: Whether to quantize the ONNX model (requires include_onnx=True)
            token: HuggingFace Hub authentication token (or set HF_TOKEN env var)
            commit_message: Commit message for the push
            private: Whether to create a private repository
            **kwargs: Additional arguments passed to HfApi.upload_folder

        Examples:
            >>> classifier.push_to_hub("my-org/my-classifier")  # ONNX included by default
            >>> classifier.push_to_hub("my-org/my-classifier", quantize_onnx=True)
            >>> classifier.push_to_hub("my-org/my-classifier", include_onnx=False)  # Opt-out
        """
        pass

    # Keep existing save/load methods for backwards compatibility
    def save(self, save_dir: str, include_onnx: bool = True, quantize_onnx: bool = True):
        """Legacy save method for backwards compatibility.

        Args:
            save_dir: Directory to save to
            include_onnx: Whether to include ONNX export (default: True)
            quantize_onnx: Whether to quantize ONNX model
        """
        pass

    @classmethod
    def load(cls, save_dir: str, device: Optional[str] = None, use_onnx: Optional[Union[bool, str]] = "auto", prefer_quantized: bool = True, trust_remote_code: bool = False) -> 'AdaptiveClassifier':
        """Legacy load method for backwards compatibility.

        Args:
            save_dir: Directory to load from
            device: Device to load model on
            use_onnx: Whether to use ONNX Runtime ("auto", True, False)
            prefer_quantized: Use quantized ONNX model if available (default: True)
            trust_remote_code: Whether to trust remote code when loading models (default: False)
        """
        pass
    
    def to(self, device: str) -> 'AdaptiveClassifier':
        """Move the model to specified device.
        
        Args:
            device: Device to move to ("cuda" or "cpu")
            
        Returns:
            Self for chaining
        """
        pass
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        pass
    
    def _initialize_adaptive_head(self):
        """Initialize or reinitialize the adaptive head with improved configuration."""
        pass

    def _get_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        """Get embeddings for input texts."""
        pass

    def get_example_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored examples and model state."""
        pass

    def predict_batch(
        self,
        texts: List[str],
        k: int = 5,
        batch_size: int = 32
    ) -> List[List[Tuple[str, float]]]:
        """Predict labels for a batch of texts with improved batching."""
        pass

    def clear_memory(self, labels: Optional[List[str]] = None):
        """Clear memory for specified labels or all if none specified."""
        pass

    def merge_classifiers(self, other: 'AdaptiveClassifier') -> 'AdaptiveClassifier':
        """Merge another classifier into this one."""
        pass
    
    def _train_adaptive_head(self, epochs: int = 10):
        """Train the adaptive head with improved stability."""
        pass
    
    def _update_adaptive_head(self):
        """Update adaptive head for new classes."""
        pass

    def select_representative_examples(self, examples: List[Example], k: int = 5) -> List[Example]:
        """Select k most representative examples using k-means clustering.
        
        Args:
            examples: List of examples to select from
            k: Number of examples to select
            
        Returns:
            List of selected examples
        """
        pass
    
    def _initialize_strategic_components(self):
        """Initialize strategic classification components."""
        pass
    
    @property
    def strategic_mode(self) -> bool:
        """Check if strategic mode is enabled and properly initialized."""
        pass
    
    def _strategic_training_step(self, all_embeddings: torch.Tensor, all_labels: torch.Tensor):
        """Perform strategic training step."""
        pass
    
    def predict_strategic(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Predict assuming the input might be strategically modified.
        
        This method simulates how a strategic agent might modify the input
        to get a better classification outcome, then predicts on that modified input.
        
        Args:
            text: Input text to classify
            k: Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples for strategic predictions
        """
        pass
    
    def predict_robust(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """Predict assuming input has already been strategically modified.
        
        This method assumes the input text has already been strategically manipulated
        and applies robust prediction techniques that are less susceptible to such manipulation.
        
        Args:
            text: Input text (potentially strategically modified)
            k: Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples for robust predictions
        """
        pass
    
    def _predict_from_embedding(
        self, 
        embedding: torch.Tensor, 
        k: int = 5, 
        robust: bool = False,
        strategic: bool = False
    ) -> List[Tuple[str, float]]:
        """Helper method to predict from embedding with strategic considerations.
        
        Args:
            embedding: Input embedding tensor
            k: Number of top predictions to return
            robust: If True, applies robust prediction weights
            strategic: If True, indicates this is for strategic prediction
            
        Returns:
            List of (label, confidence) tuples
        """
        pass
    
    def evaluate_strategic_robustness(
        self,
        test_texts: List[str],
        test_labels: List[str],
        gaming_levels: List[float] = [0.0, 0.5, 1.0]
    ) -> Dict[str, float]:
        """Evaluate strategic robustness of the classifier."""
        pass
