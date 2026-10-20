# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Verify repeated large activations through an exact column-select MatMulNBits."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort


class TestWebGpuActivationLayout(unittest.TestCase):
    def test_repeated_large_activation_copy_preserves_every_row(self):
        library = os.environ.get("ORT_WEBGPU_TEST_LIBRARY")
        if not library:
            self.skipTest("Set ORT_WEBGPU_TEST_LIBRARY to a native WebGPU plugin")
        dll_directory = os.add_dll_directory(str(Path(library).resolve().parent)) if os.name == "nt" else None
        ort.register_execution_provider_library("WebGpuExecutionProvider", library)
        devices = [d for d in ort.get_ep_devices() if d.ep_name == "WebGpuExecutionProvider"]
        self.assertTrue(devices, "No native WebGPU device")
        m, n, k = 3072, 128, 4096
        # Select the first N columns exactly. Nonzero, distinct row patterns expose
        # missing four-row bands without a tolerance obscuring the copy failure.
        a = ((np.arange(m)[:, None] * 17 + np.arange(k)[None, :] * 13) % 509 - 254).astype(np.float16)
        a /= np.float16(256)
        q = np.full((n, k), 8, np.uint8)
        q[np.arange(n), np.arange(n)] = 9
        packed = (q[:, ::2] | (q[:, 1::2] << 4)).reshape(n, k // 32, 16)
        graph = helper.make_graph(
            [
                helper.make_node(
                    "MatMulNBits",
                    ["a", "b", "scale", "zero"],
                    ["y"],
                    name="column_select",
                    domain="com.microsoft",
                    K=k,
                    N=n,
                    bits=4,
                    block_size=32,
                    accuracy_level=4,
                )
            ],
            "activation_layout",
            [helper.make_tensor_value_info("a", TensorProto.FLOAT16, [m, k])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [m, n])],
            [
                numpy_helper.from_array(packed, "b"),
                numpy_helper.from_array(np.ones((n, k // 32), np.float16), "scale"),
                numpy_helper.from_array(np.full((n, k // 32), 8, np.float16), "zero"),
            ],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)], ir_version=10
        )
        with tempfile.TemporaryDirectory() as directory:
            options = ort.SessionOptions()
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(directory) / "layout")
            options.add_provider_for_devices(
                devices[:1],
                {
                    "dawnBackendType": os.environ.get("ORT_WEBGPU_TEST_BACKEND", "Vulkan"),
                    "enableGraphCapture": "0",
                },
            )
            session = ort.InferenceSession(model.SerializeToString(), options)
            binding = session.io_binding()
            source = session.create_ortvalue_from_shape_and_type(a.shape, a.dtype, "webgpu")
            target = session.create_ortvalue_from_shape_and_type((m, n), a.dtype, "webgpu")
            source.update_inplace(a)
            binding.bind_ortvalue_input("a", source)
            binding.bind_ortvalue_output("y", target)
            failures = []
            for iteration in range(int(os.environ.get("ORT_WEBGPU_TEST_REPEATS", "4096"))):
                session.run_with_iobinding(binding)
                actual = target.numpy()
                if not np.array_equal(actual, a[:, :n]):
                    failures.append(iteration)
            events = json.loads(Path(session.end_profiling()).read_text())
            self.assertFalse(any(e.get("args", {}).get("provider") == "CPUExecutionProvider" for e in events))
            shaders = [e for e in events if e.get("cat") == "Api" and "cache_key" in e.get("args", {})]
            if not any("&SubgroupMatrixMatMulNBits" in e["name"] for e in shaders):
                self.skipTest("This adapter does not dispatch the subgroup-matrix path")
            self.assertFalse(failures, f"Column-select failed {len(failures)} repeats; first failures: {failures[:16]}")
        if dll_directory:
            dll_directory.close()


if __name__ == "__main__":
    unittest.main()
