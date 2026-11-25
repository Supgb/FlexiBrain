import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from flexibrain.cli import parse_args
from flexibrain.config import RunConfig
from flexibrain.engine.downstream_trainer import DownstreamTrainer
from flexibrain.engine.pretrainer import Pretrainer
from flexibrain.utils.training import get_dynamic_momentum


class TinyPretrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.target_encoder = nn.Linear(4, 4)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x, **kwargs):
        return x


class SchedulerAndSelectionTests(unittest.TestCase):
    def test_get_dynamic_momentum_linear_and_cosine(self):
        self.assertAlmostEqual(get_dynamic_momentum(0, 100, 0.996, 0.9999, schedule="linear"), 0.996, places=9)
        self.assertAlmostEqual(get_dynamic_momentum(99, 100, 0.996, 0.9999, schedule="linear"), 0.9999, places=9)
        mid_lin = get_dynamic_momentum(50, 101, 0.996, 0.9999, schedule="linear")
        self.assertAlmostEqual(mid_lin, 0.5 * (0.996 + 0.9999), places=9)

        self.assertAlmostEqual(get_dynamic_momentum(0, 100, 0.996, 0.9999, schedule="cosine"), 0.996, places=9)
        self.assertAlmostEqual(get_dynamic_momentum(99, 100, 0.996, 0.9999, schedule="cosine"), 0.9999, places=9)
        with self.assertRaises(ValueError):
            get_dynamic_momentum(0, 10, schedule="exponential")

    def test_pretrainer_scheduler_accounts_for_grad_accum_and_single_cosine_cycle(self):
        cfg = RunConfig()
        cfg.training.epochs = 10
        cfg.training.warmup_epochs = 2
        cfg.training.lr = 5e-4
        cfg.training.grad_accumulation_steps = 4
        cfg.training.use_amp = False

        fake_train_loader = [None] * 8  # 8 micro-batches / 4 accum = 2 optimizer steps/epoch -> 20 total, 4 warmup
        fake_val_loader = [None]
        model = TinyPretrainModel()

        with (
            patch("flexibrain.engine.pretrainer.setup_logger"),
            patch("flexibrain.engine.pretrainer.torch.cuda.is_available", return_value=False),
            patch("flexibrain.engine.pretrainer.build_pretrain_model", return_value=model),
            patch("flexibrain.engine.pretrainer.build_pretrain_dataloaders", return_value=(fake_train_loader, fake_val_loader)),
        ):
            trainer = Pretrainer(cfg).build()

        self.assertEqual(trainer.optimizer.defaults["betas"], (0.9, 0.999))
        lrs = []
        for _ in range(20):
            lrs.append(trainer.optimizer.param_groups[0]["lr"])
            trainer.optimizer.step()
            trainer.scheduler.step()

        # Warmup over first 4 optimizer steps: 0.25, 0.50, 0.75, 1.00 of peak LR
        expected_warmup = [5e-4 * (i + 1) / 4 for i in range(4)]
        for got, exp in zip(lrs[:4], expected_warmup):
            self.assertAlmostEqual(got, exp, places=10)

        # Single monotonic cosine decay from step 4 to step 19 (no premature zero restarts)
        decay_lrs = lrs[4:]
        self.assertTrue(all(a > b > 0.0 for a, b in zip(decay_lrs, decay_lrs[1:])))
        self.assertAlmostEqual(decay_lrs[0], 5e-4, places=10)

    def test_pretrainer_amp_overflow_skips_scheduler_and_ema(self):
        cfg = RunConfig()
        cfg.training.use_amp = True
        cfg.training.grad_clip = 0.0
        with patch("flexibrain.engine.pretrainer.setup_logger"), patch(
            "flexibrain.engine.pretrainer.torch.cuda.is_available", return_value=False
        ):
            trainer = Pretrainer(cfg)

        trainer.use_amp = True
        trainer.model = TinyPretrainModel()
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
        trainer.scheduler = MagicMock()

        trainer.scaler = MagicMock()
        trainer.scaler.get_scale.side_effect = [65536.0, 32768.0]  # Overflow lowered scale
        with patch("flexibrain.engine.pretrainer.update_ema") as upd:
            trainer._optimizer_step(0.996)
        upd.assert_not_called()
        trainer.scheduler.step.assert_not_called()

        trainer.scaler.get_scale.side_effect = [32768.0, 32768.0]  # Valid step
        with patch("flexibrain.engine.pretrainer.update_ema") as upd:
            trainer._optimizer_step(0.996)
        upd.assert_called_once()
        trainer.scheduler.step.assert_called_once()

    def test_downstream_scheduler_accounts_for_grad_accum(self):
        cfg = RunConfig()
        cfg.training.epochs = 10
        cfg.training.warmup_epochs = 2
        cfg.training.lr = 1e-5
        cfg.training.grad_accumulation_steps = 4
        cfg.training.use_amp = False

        fake_train_loader = [None] * 12  # 3 optimizer steps per epoch -> 30 total, 6 warmup
        model = TinyClassifier()
        with (
            patch("flexibrain.engine.downstream_trainer.setup_logger"),
            patch("flexibrain.engine.downstream_trainer.torch.cuda.is_available", return_value=False),
            patch("flexibrain.engine.downstream_trainer.build_downstream_model", return_value=model),
            patch(
                "flexibrain.engine.downstream_trainer.build_downstream_dataloaders",
                return_value=(fake_train_loader, [None], None),
            ),
        ):
            trainer = DownstreamTrainer(cfg).build()

        self.assertEqual(trainer.optimizer.defaults["betas"], (0.9, 0.999))
        lrs = []
        for _ in range(30):
            lrs.append(trainer.optimizer.param_groups[0]["lr"])
            trainer.optimizer.step()
            trainer.scheduler.step()

        self.assertAlmostEqual(lrs[0], 1e-5 / 6.0, places=11)
        self.assertAlmostEqual(lrs[5], 1e-5, places=11)
        self.assertAlmostEqual(lrs[6], 1e-5, places=11)
        self.assertTrue(all(a > b > 0.0 for a, b in zip(lrs[6:], lrs[7:])))
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.0, places=11)

    def test_evaluate_sample_weighted_loss_auroc_and_balanced_acc(self):
        with patch("flexibrain.engine.downstream_trainer.setup_logger"):
            trainer = DownstreamTrainer(RunConfig())
        trainer.device = torch.device("cpu")
        trainer.use_amp = False
        trainer.model = TinyClassifier()

        b1 = (torch.tensor([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0]]), None, None, torch.tensor([0, 0, 1]), None)
        b2 = (torch.tensor([[0.0, 3.0]]), None, None, torch.tensor([1]), None)
        ce = nn.CrossEntropyLoss(reduction="sum")
        expected_loss = float((ce(b1[0], b1[3]) + ce(b2[0], b2[3])).item() / 4.0)

        with patch("flexibrain.engine.downstream_trainer.prepare_batch_data", side_effect=[b1, b2]):
            metrics = trainer.evaluate([None, None], "Validation")

        self.assertAlmostEqual(metrics["loss"], expected_loss, places=6)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["auroc"], 1.0)

        # Single-class batch yields auroc=None cleanly
        single_cls = (torch.tensor([[2.0, 0.0], [1.0, 0.0]]), None, None, torch.tensor([0, 0]), None)
        with patch("flexibrain.engine.downstream_trainer.prepare_batch_data", return_value=single_cls):
            metrics_single = trainer.evaluate([None], "Validation")
        self.assertIsNone(metrics_single["auroc"])

        # Multiclass OVR AUROC
        b_mc = (
            torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),
            None,
            None,
            torch.tensor([0, 1, 2]),
            None,
        )
        with patch("flexibrain.engine.downstream_trainer.prepare_batch_data", return_value=b_mc):
            metrics_mc = trainer.evaluate([None], "Validation")
        self.assertEqual(metrics_mc["auroc"], 1.0)

    def test_downstream_selection_metrics_and_auroc_fallback(self):
        def run_with_val_sequence(selection_metric: str, val_seq: list[dict]):
            with tempfile.TemporaryDirectory() as tmpdir:
                cfg = RunConfig()
                cfg.training.epochs = len(val_seq)
                cfg.training.selection_metric = selection_metric
                cfg.logging.checkpoint_dir = tmpdir
                with patch("flexibrain.engine.downstream_trainer.setup_logger"), patch(
                    "flexibrain.engine.downstream_trainer.torch.cuda.is_available", return_value=False
                ):
                    trainer = DownstreamTrainer(cfg)
                trainer.model = TinyClassifier()
                trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-5)
                trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
                trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
                trainer.train_loader = SimpleNamespace(dataset=[0])
                trainer.val_loader = SimpleNamespace(dataset=[0])
                trainer.test_loader = None

                save_calls = []
                orig_save = trainer.save

                def record_save(*args, **kwargs):
                    save_calls.append((args[0], kwargs["is_best"], kwargs["best_score"]))
                    return orig_save(*args, **kwargs)

                with (
                    patch.object(trainer, "build", return_value=trainer),
                    patch.object(trainer, "train_one_epoch", return_value=0.6),
                    patch.object(trainer, "evaluate", side_effect=val_seq),
                    patch.object(trainer, "save", side_effect=record_save),
                ):
                    trainer.fit()
                best_ckpt = torch.load(Path(tmpdir) / "downstream_best.pt", map_location="cpu", weights_only=True)
                return save_calls, best_ckpt["epoch"]

        # 1. val_auroc wins over noisier f1_macro
        seq1 = [
            {"auroc": 0.60, "loss": 0.69, "f1_macro": 0.40},
            {"auroc": 0.68, "loss": 0.67, "f1_macro": 0.50},
            {"auroc": 0.62, "loss": 0.72, "f1_macro": 0.75},  # higher F1 but worse AUROC & loss
        ]
        calls1, best_ep1 = run_with_val_sequence("val_auroc", seq1)
        self.assertEqual([c[1] for c in calls1], [True, True, False])
        self.assertEqual(best_ep1, 1)

        # 2. val_loss selects minimum CE loss
        calls2, best_ep2 = run_with_val_sequence("val_loss", seq1)
        self.assertEqual([c[1] for c in calls2], [True, True, False])
        self.assertEqual(best_ep2, 1)

        # 3. val_auroc with initial None falls back to -val_loss, then yields to first real AUROC
        seq3 = [
            {"auroc": None, "loss": 0.80, "f1_macro": 0.2},
            {"auroc": None, "loss": 0.70, "f1_macro": 0.2},
            {"auroc": 0.52, "loss": 0.75, "f1_macro": 0.3},
        ]
        calls3, best_ep3 = run_with_val_sequence("val_auroc", seq3)
        self.assertEqual([c[1] for c in calls3], [True, True, True])
        self.assertEqual(best_ep3, 2)

        # 4. Invalid selection_metric raises
        cfg_bad = RunConfig()
        cfg_bad.training.selection_metric = "invalid_metric"
        with patch("flexibrain.engine.downstream_trainer.setup_logger"):
            bad_trainer = DownstreamTrainer(cfg_bad)
        with self.assertRaises(ValueError):
            bad_trainer.fit()

    def test_cli_flags_parse(self):
        with patch.object(sys, "argv", ["flexibrain", "pretrain", "--config", "x.yaml", "--ema-schedule", "cosine"]):
            args = parse_args()
        self.assertEqual(args.ema_schedule, "cosine")

        with patch.object(sys, "argv", ["flexibrain", "downstream", "--config", "x.yaml", "--selection-metric", "val_loss"]):
            args_ds = parse_args()
        self.assertEqual(args_ds.selection_metric, "val_loss")


if __name__ == "__main__":
    unittest.main()
