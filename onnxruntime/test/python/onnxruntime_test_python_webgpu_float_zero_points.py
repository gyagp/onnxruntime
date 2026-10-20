# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Native MatMulNBits fractional zero-point regression tests.

Run with ORT_WEBGPU_TEST_LIBRARY pointing to the built plugin and its dependency
DLLs beside it. ORT_WEBGPU_TEST_BACKEND selects D3D12 or Vulkan. The test checks
the executed operator and rejects CPU fallback; inputs are not constant-folded.
"""

import gc
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort


class TestWebGpuFloatZeroPoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = os.environ.get("ORT_WEBGPU_TEST_LIBRARY")
        if not library:
            raise unittest.SkipTest("Set ORT_WEBGPU_TEST_LIBRARY to a native WebGPU plugin")
        cls.dll_directory = os.add_dll_directory(str(Path(library).resolve().parent)) if os.name == "nt" else None
        ort.register_execution_provider_library("WebGpuExecutionProvider", library)
        cls.devices = [d for d in ort.get_ep_devices() if d.ep_name == "WebGpuExecutionProvider"]
        if not cls.devices:
            raise RuntimeError("Native WebGPU device not available")
        cls.backend = os.environ.get("ORT_WEBGPU_TEST_BACKEND", "D3D12")
        cls.records = []

    def check_case(self, m, n, k, bits, dtype, block=32, packed_zero=False, capture=False, batch=1, bias=False):
        rng = np.random.default_rng(9182)
        count = (k + block - 1) // block
        center = 1 << (bits - 1)
        q = (center + rng.integers(-4, 5, (n, count, block))).astype(np.uint8)
        packed = q if bits == 8 else q[..., ::2] | (q[..., 1::2] << 4)
        zero = (center + 0.3125 + (np.arange(n)[:, None] % 4) * 0.125 + np.arange(count)[None] * 0.0625).astype(dtype)
        if packed_zero:
            zero = np.round(zero).astype(dtype)
            if bits == 4:
                z = np.pad(zero.astype(np.uint8), ((0, 0), (0, count % 2)))
                zero_feed = z[:, ::2] | (z[:, 1::2] << 4)
            else:
                zero_feed = zero.astype(np.uint8)
        else:
            zero_feed = zero
        scale = (0.015625 + (np.arange(n)[:, None] % 3) / 512 + np.arange(count)[None] / 1024).astype(dtype)
        a = rng.uniform(0.4, 0.7, (m, k) if batch == 1 else (batch, m, k)).astype(dtype)
        weight = (
            (q.astype(np.float32) - zero.astype(np.float32)[:, :, None]) * scale.astype(np.float32)[:, :, None]
        ).reshape(n, -1)[:, :k]
        expected = a.astype(np.float32) @ weight.T
        bias_values = np.linspace(-0.5, 0.5, n, dtype=dtype) if bias else None
        if bias:
            expected += bias_values.astype(np.float32)
        element = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
        node = helper.make_node(
            "MatMulNBits",
            ["a", "b", "scales", "zero", "", "bias"] if bias else ["a", "b", "scales", "zero"],
            ["y"],
            name="fractional_zero",
            domain="com.microsoft",
            N=n,
            K=k,
            bits=bits,
            block_size=block,
            accuracy_level=4,
        )
        graph = helper.make_graph(
            [node],
            "fractional_zero",
            [helper.make_tensor_value_info("a", element, list(a.shape))],
            [helper.make_tensor_value_info("y", element, list(expected.shape))],
            [
                numpy_helper.from_array(packed, "b"),
                numpy_helper.from_array(scale, "scales"),
                numpy_helper.from_array(zero_feed, "zero"),
            ]
            + ([numpy_helper.from_array(bias_values, "bias")] if bias else []),
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)], ir_version=10
        )
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.log_severity_level = 1
        options.enable_profiling = True
        with tempfile.TemporaryDirectory() as directory:
            options.profile_file_prefix = str(Path(directory) / "profile")
            options.add_provider_for_devices(
                self.devices[:1],
                {
                    "dawnBackendType": self.backend,
                    "validationMode": "full",
                    "enableGraphCapture": "1" if capture else "0",
                },
            )
            session = ort.InferenceSession(model.SerializeToString(), options)
            session.disable_fallback()
            if capture:
                binding = session.io_binding()
                input_value = session.create_ortvalue_from_shape_and_type(a.shape, a.dtype, "webgpu")
                output_value = session.create_ortvalue_from_shape_and_type(expected.shape, a.dtype, "webgpu")
                binding.bind_ortvalue_input("a", input_value)
                binding.bind_ortvalue_output("y", output_value)
                for iteration in range(20):
                    sample = (a * (1 + iteration / 100)).astype(dtype)
                    input_value.update_inplace(sample)
                    session.run_with_iobinding(binding)
                    actual = output_value.numpy().astype(np.float32)
                    expected = sample.astype(np.float32) @ weight.T
                    if bias:
                        expected += bias_values.astype(np.float32)
                    np.testing.assert_allclose(actual, expected, atol=0.035, rtol=0.015)
                session.release_captured_graph(0)
                binding.clear_binding_inputs()
                binding.clear_binding_outputs()
                del binding, input_value, output_value
            else:
                actual = session.run(None, {"a": a})[0].astype(np.float32)
            np.testing.assert_allclose(
                actual,
                expected,
                atol=0.035 if dtype == np.float16 else 0.001,
                rtol=0.015 if dtype == np.float16 else 0.0005,
            )
            events = json.loads(Path(session.end_profiling()).read_text())
            kernels = [e for e in events if e.get("cat") == "Node" and e.get("args", {}).get("provider")]
            operators = [e for e in kernels if e["args"]["op_name"] == "MatMulNBits"]
            self.assertGreaterEqual(len(operators), 1)
            if capture:
                self.assertEqual(sum(e.get("name") == "model_run" for e in events), 20)
                self.assertLess(sum(e.get("name") == "SequentialExecutor::Execute" for e in events), 20)
            self.assertTrue(all(e["args"]["provider"] == "WebGpuExecutionProvider" for e in kernels))
            self.records.append(
                {
                    "backend": self.backend,
                    "m": m,
                    "n": n,
                    "k": k,
                    "bits": bits,
                    "dtype": str(dtype),
                    "block": block,
                    "packed_zero": packed_zero,
                    "capture": capture,
                    "batch": batch,
                    "bias": bias,
                    "executions": 20 if capture else 1,
                    "max_error": float(np.max(np.abs(actual - expected))),
                }
            )
            del session
            gc.collect()

    def test_fractional_zero_points_and_row_strides(self):
        for shape in [(1, 17, 80), (3, 17, 80), (16, 64, 64), (128, 128, 128)]:
            for bits in (4, 8):
                for dtype in (np.float16, np.float32):
                    with self.subTest(shape=shape, bits=bits, dtype=dtype):
                        self.check_case(*shape, bits, dtype)

    def test_packed_zero_points_are_unchanged(self):
        for bits in (4, 8):
            for dtype in (np.float16, np.float32):
                with self.subTest(bits=bits, dtype=dtype):
                    self.check_case(16, 64, 80, bits, dtype, packed_zero=True)

    def test_block16_float_zero_points(self):
        for bits in (4, 8):
            self.check_case(3, 17, 48, bits, np.float16, block=16)

    def test_tall_tiles_and_partial_k_guard(self):
        for k in (256, 288):
            for bits in (4, 8):
                for packed in (False, True):
                    with self.subTest(k=k, bits=bits, packed=packed):
                        self.check_case(513, 256, k, bits, np.float16, packed_zero=packed)

    def test_capture_replays_with_padded_rows(self):
        for m in (129, 257, 513):
            for bits in (4, 8):
                with self.subTest(m=m, bits=bits):
                    self.check_case(m, 128, 256, bits, np.float16, capture=True)

    def test_wide_tile_bias_and_batch_with_partial_rows_and_columns(self):
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=dtype):
                self.check_case(33, 131, 256, 4, dtype, batch=2, bias=True)

    @classmethod
    def tearDownClass(cls):
        path = os.environ.get("ORT_WEBGPU_TEST_REPORT")
        if path:
            Path(path).write_text(json.dumps(cls.records, indent=2))


if __name__ == "__main__":
    unittest.main()
