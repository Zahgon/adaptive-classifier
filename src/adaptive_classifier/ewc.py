import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import copy

class EWC:
    """Elastic Weight Consolidation for preventing catastrophic forgetting."""
    
    def __init__(
        self,
        model: nn.Module,
        dataset: torch.utils.data.Dataset,
        device: str = 'cpu',
        ewc_lambda: float = 100.0
    ):
        """Initialize EWC.
        
        Args:
            model: Neural network model
            dataset: Dataset to compute Fisher information
            device: Device to use
            ewc_lambda: Importance of old tasks
        """
        pass
    
    def _compute_fisher(
        self,
        dataset: torch.utils.data.Dataset
    ) -> Dict[str, torch.Tensor]:
        """Compute Fisher information matrix.
        
        Args:
            dataset: Dataset to compute Fisher information
            
        Returns:
            Dictionary of parameter names to Fisher information
        """
        pass
    
    def ewc_loss(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Compute EWC loss.
        
        Args:
            batch_size: Batch size for normalization
            
        Returns:
            EWC loss tensor
        """
        pass
