import torch
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class SRMetricSuite:
    def __init__(self, device):
        metrics = MetricCollection({
            'psnr': PeakSignalNoiseRatio(data_range=1.0),
            'ssim': StructuralSimilarityIndexMeasure(data_range=1.0)
        })
        self.collection = metrics.to(device)

    def update(self, preds, targets):
        """
        Standardize ranges before updating
        """        
        self.collection.update(preds, targets)

    def compute(self):
        # Returns a dictionary: {'psnr': tensor, 'ssim': tensor}
        return self.collection.compute()

    def reset(self):
        self.collection.reset()