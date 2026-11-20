import unittest

import torch
import torch.nn.functional as F

from flexibrain.models.mamba_blocks import Mamba, Mamba2
from flexibrain._mamba.ops.selective_scan_interface import selective_scan_ref


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class MambaCudaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_bimamba_v2_matches_reference_gradients(self):
        model = Mamba(32, bimamba_type="v2", if_devide_out=True, device="cuda")
        inputs = torch.randn(2, 11, 32, device="cuda", requires_grad=True)
        actual = model(inputs)

        xz = model.in_proj(inputs).transpose(1, 2)

        def scan(xz, conv, x_proj, dt_proj, A_log, D):
            x, z = xz.chunk(2, dim=1)
            conv_out = conv(x)[..., :x.shape[-1]]
            dt, B, C = x_proj(conv_out.transpose(1, 2)).split(
                [model.dt_rank, model.d_state, model.d_state], dim=-1
            )
            delta = F.linear(dt, dt_proj.weight).transpose(1, 2)
            return selective_scan_ref(
                conv_out, delta, -A_log.exp(), B.transpose(1, 2), C.transpose(1, 2),
                D, z, dt_proj.bias, delta_softplus=True,
            )

        forward = scan(xz, model.conv1d, model.x_proj, model.dt_proj, model.A_log, model.D)
        reverse = scan(
            xz.flip([-1]), model.conv1d_b, model.x_proj_b, model.dt_proj_b, model.A_b_log, model.D_b
        ).flip([-1])
        expected = model.out_proj((forward + reverse).transpose(1, 2) / 2)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
        parameters = (inputs, *model.parameters())
        actual_grads = torch.autograd.grad(actual.square().mean(), parameters)
        expected_grads = torch.autograd.grad(expected.square().mean(), parameters)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-4)

    def test_other_mamba_mixers_support_backward(self):
        for model in (
            Mamba(32, bimamba_type="none", device="cuda"),
            Mamba(32, bimamba_type="v1", device="cuda"),
            Mamba2(64, d_state=32, headdim=32, chunk_size=32, device="cuda"),
        ):
            with self.subTest(mixer=type(model).__name__, bimamba_type=getattr(model, "bimamba_type", None)):
                inputs = torch.randn(2, 11, model.d_model, device="cuda", requires_grad=True)
                output = model(inputs)
                self.assertEqual(output.shape, inputs.shape)
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())
                self.assertTrue(torch.isfinite(model.conv1d.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
