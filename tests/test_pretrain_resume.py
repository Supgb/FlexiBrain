import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch
from torch import nn

from flexibrain.config import RunConfig
from flexibrain.engine.pretrainer import Pretrainer


class PretrainResumeTests(unittest.TestCase):
    def test_fit_restores_state_and_resumes_after_checkpoint_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint_latest.pt"
            saved_model = nn.Linear(1, 1, bias=False)
            saved_optimizer = torch.optim.SGD(saved_model.parameters(), lr=0.1)
            saved_scheduler = torch.optim.lr_scheduler.LambdaLR(saved_optimizer, lambda _: 1.0)
            with torch.no_grad():
                saved_model.weight.fill_(7.0)
            torch.save(
                {
                    "epoch": 2,
                    "model_state_dict": saved_model.state_dict(),
                    "optimizer_state_dict": saved_optimizer.state_dict(),
                    "scheduler_state_dict": saved_scheduler.state_dict(),
                    "scaler_state_dict": {"saved": True},
                    "val_loss": 0.5,
                    "best_loss": 0.5,
                    "config": {},
                },
                checkpoint_path,
            )

            cfg = RunConfig()
            cfg.training.epochs = 5
            cfg.logging.resume = str(checkpoint_path)
            with patch("flexibrain.engine.pretrainer.setup_logger"), patch(
                "flexibrain.engine.pretrainer.torch.cuda.is_available", return_value=False
            ):
                trainer = Pretrainer(cfg)

            trainer.model = nn.Linear(1, 1, bias=False)
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
            trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
            trainer.scaler = MagicMock()
            trainer.train_loader = SimpleNamespace(sampler=SimpleNamespace(), dataset=[0])
            trainer.val_loader = SimpleNamespace(dataset=[0])

            with (
                patch.object(trainer, "build", return_value=trainer),
                patch.object(trainer, "train_one_epoch", return_value=0.6) as train,
                patch.object(trainer, "validate", return_value=0.6) as validate,
                patch.object(trainer, "save") as save,
            ):
                trainer.fit()

            self.assertEqual(train.call_args_list, [call(3), call(4)])
            self.assertEqual(validate.call_args_list, [call(3), call(4)])
            self.assertEqual(
                save.call_args_list,
                [call(3, 0.6, 0.5, is_best=False), call(4, 0.6, 0.5, is_best=False)],
            )
            torch.testing.assert_close(trainer.model.weight, saved_model.weight)
            trainer.scaler.load_state_dict.assert_called_once_with({"saved": True})


if __name__ == "__main__":
    unittest.main()
