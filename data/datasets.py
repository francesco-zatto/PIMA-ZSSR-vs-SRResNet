import torch
from torch.utils.data import Dataset
from data.preprocessing import SRPreprocessingStrategy
from abc import ABC

class AbstractSRDataset(Dataset, ABC):
    """
    Generic SISR Dataset that delegate preprocessing operations to a SRPreprocessingStrategy.
    """
    def __init__(self, root_dir: str, scale_factor: float, strategy: SRPreprocessingStrategy, ext: str):
        self.scale_factor = scale_factor
        self.strategy = strategy

        self.strategy.prepare(root_dir, ext)

    def __len__(self) -> int:
        return len(self.strategy)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._check(self.strategy.sample(idx, self.scale_factor))
    
    def _check(self, item: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply final check to LR and HR images, i.e. clamping them between 0.0 and 1.0.
        """
        lr_img, hr_img = item
        return torch.clamp(lr_img, 0.0, 1.0), torch.clamp(hr_img, 0.0, 1.0)
    

class Urban100Dataset(AbstractSRDataset):
    def __init__(self, root_dir: str, scale_factor: float, strategy: SRPreprocessingStrategy):
        super().__init__(root_dir, scale_factor, strategy, ext="*.png")

class BSD100Dataset(AbstractSRDataset):
    def __init__(self, root_dir: str, scale_factor: float, strategy: SRPreprocessingStrategy):
        super().__init__(root_dir, scale_factor, strategy, ext="*.png")