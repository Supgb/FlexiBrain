import io
import unittest
from unittest.mock import patch

import torch
from torch import nn

from flexibrain.config import RunConfig
from flexibrain.engine.downstream_trainer import DownstreamTrainer


class IdentityClassifier(nn.Module):
    def forward(self, x, **kwargs):
        return x


class DownstreamCheckpointTests(unittest.TestCase):
    def test_validation_metrics_can_be_loaded_with_weights_only(self):
        with patch("flexibrain.engine.downstream_trainer.setup_logger"):
            trainer = DownstreamTrainer(RunConfig())
        trainer.device = torch.device("cpu")
        trainer.use_amp = False
        trainer.model = IdentityClassifier()
        logits = torch.tensor([[2.0, 1.0], [2.0, 3.0]])
        labels = torch.tensor([0, 1])
        batch = logits, None, None, labels, None
        with patch("flexibrain.engine.downstream_trainer.prepare_batch_data", return_value=batch):
            metrics = trainer.evaluate([None], "Validation")

        checkpoint = io.BytesIO()
        torch.save({"metrics": metrics}, checkpoint)
        checkpoint.seek(0)
        restored = torch.load(checkpoint, weights_only=True)
        self.assertEqual(restored["metrics"], metrics)
        self.assertEqual(restored["metrics"]["accuracy"], 1.0)
