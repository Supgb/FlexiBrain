import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch
from torch import nn

from flexibrain.config import DataConfig, RunConfig, TrainingConfig
from flexibrain.data.builders import build_downstream_dataloaders
from flexibrain.engine.downstream_trainer import DownstreamTrainer


class DownstreamResumeTests(unittest.TestCase):
    def test_fit_restores_state_and_best_f1_from_existing_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            latest_path = Path(tmpdir) / "downstream_latest.pt"
            best_path = Path(tmpdir) / "downstream_best.pt"
            saved_model = nn.Linear(1, 1, bias=False)
            saved_optimizer = torch.optim.SGD(saved_model.parameters(), lr=0.1)
            saved_scheduler = torch.optim.lr_scheduler.LambdaLR(saved_optimizer, lambda _: 1.0)
            with torch.no_grad():
                saved_model.weight.fill_(7.0)
            payload = {
                "epoch": 15,
                "model": saved_model.state_dict(),
                "optimizer": saved_optimizer.state_dict(),
                "scheduler": saved_scheduler.state_dict(),
                "scaler": {"saved": True},
                "metrics": {"val": {"f1_macro": 0.5}, "train_loss": 0.68},
                "config": {},
            }
            torch.save(payload, latest_path)
            best_payload = dict(payload)
            best_payload["epoch"] = 13
            best_payload["metrics"] = {"val": {"f1_macro": 0.578018}, "train_loss": 0.69}
            torch.save(best_payload, best_path)

            cfg = RunConfig()
            cfg.training.epochs = 18
            cfg.training.selection_metric = "f1_macro"
            cfg.logging.resume = str(latest_path)
            cfg.logging.checkpoint_dir = tmpdir
            with patch("flexibrain.engine.downstream_trainer.setup_logger"), patch(
                "flexibrain.engine.downstream_trainer.torch.cuda.is_available", return_value=False
            ):
                trainer = DownstreamTrainer(cfg)

            trainer.model = nn.Linear(1, 1, bias=False)
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
            trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
            trainer.scaler = MagicMock()
            trainer.train_loader = SimpleNamespace(sampler=SimpleNamespace(), dataset=[0])
            trainer.val_loader = SimpleNamespace(dataset=[0])
            trainer.test_loader = None

            with (
                patch.object(trainer, "build", return_value=trainer),
                patch.object(trainer, "train_one_epoch", return_value=0.68) as train,
                patch.object(trainer, "evaluate", return_value={"f1_macro": 0.57, "loss": 0.69}) as evaluate,
                patch.object(trainer, "save") as save,
            ):
                trainer.fit()

            self.assertEqual(train.call_args_list, [call(16), call(17)])
            self.assertEqual(evaluate.call_args_list, [call(trainer.val_loader, "Validation")] * 2)
            self.assertEqual([c.args[0] for c in save.call_args_list], [16, 17])
            self.assertEqual([c.kwargs["best_score"] for c in save.call_args_list], [0.578018] * 2)
            self.assertEqual([c.kwargs["selection_metric"] for c in save.call_args_list], ["f1_macro"] * 2)
            self.assertTrue(all(c.kwargs["is_best"] is False for c in save.call_args_list))
            torch.testing.assert_close(trainer.model.weight, saved_model.weight)
            trainer.scaler.load_state_dict.assert_called_once_with({"saved": True})

    def test_fit_restores_val_auroc_best_score_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            latest_path = Path(tmpdir) / "downstream_latest.pt"
            saved_model = nn.Linear(1, 1, bias=False)
            saved_optimizer = torch.optim.SGD(saved_model.parameters(), lr=0.1)
            saved_scheduler = torch.optim.lr_scheduler.LambdaLR(saved_optimizer, lambda _: 1.0)
            torch.save(
                {
                    "epoch": 4,
                    "model": saved_model.state_dict(),
                    "optimizer": saved_optimizer.state_dict(),
                    "scheduler": saved_scheduler.state_dict(),
                    "metrics": {"val": {"auroc": 0.65, "loss": 0.70}, "train_loss": 0.68},
                    "selection_metric": "val_auroc",
                    "best_score": 0.71,
                    "best_is_fallback": False,
                    "config": {},
                },
                latest_path,
            )

            cfg = RunConfig()
            cfg.training.epochs = 7
            cfg.training.selection_metric = "val_auroc"
            cfg.logging.resume = str(latest_path)
            cfg.logging.checkpoint_dir = tmpdir
            with patch("flexibrain.engine.downstream_trainer.setup_logger"), patch(
                "flexibrain.engine.downstream_trainer.torch.cuda.is_available", return_value=False
            ):
                trainer = DownstreamTrainer(cfg)

            trainer.model = nn.Linear(1, 1, bias=False)
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
            trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
            trainer.scaler = MagicMock()
            trainer.train_loader = SimpleNamespace(sampler=SimpleNamespace(), dataset=[0])
            trainer.val_loader = SimpleNamespace(dataset=[0])
            trainer.test_loader = None

            val_seq = [
                {"auroc": 0.69, "loss": 0.64, "f1_macro": 0.60},
                {"auroc": 0.75, "loss": 0.66, "f1_macro": 0.62},
            ]
            with (
                patch.object(trainer, "build", return_value=trainer),
                patch.object(trainer, "train_one_epoch", return_value=0.65),
                patch.object(trainer, "evaluate", side_effect=val_seq),
                patch.object(trainer, "save") as save,
            ):
                trainer.fit()

            self.assertEqual([c.kwargs["is_best"] for c in save.call_args_list], [False, True])
            self.assertEqual([c.kwargs["best_score"] for c in save.call_args_list], [0.71, 0.75])

    @patch("flexibrain.data.builders.DataLoader")
    @patch("flexibrain.data.builders._classification_dataset")
    def test_evaluation_batch_size_does_not_change_training_batch_size(self, dataset, loader):
        dataset.side_effect = [object(), object(), object()]
        loader.side_effect = [MagicMock(), MagicMock(), MagicMock()]
        data = DataConfig(
            train_list="train.txt",
            val_list="val.txt",
            test_list="test.txt",
            csv="labels.csv",
            batch_size=8,
            eval_batch_size=4,
            num_workers=0,
        )

        build_downstream_dataloaders(data, TrainingConfig())

        self.assertEqual([c.kwargs["batch_size"] for c in loader.call_args_list], [8, 4, 4])
        self.assertEqual([c.kwargs["drop_last"] for c in loader.call_args_list], [True, False, False])


if __name__ == "__main__":
    unittest.main()
