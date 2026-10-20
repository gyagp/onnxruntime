# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper

import onnxruntime as ort


class TestWebGpuPluginProfiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = os.environ.get("ORT_WEBGPU_TEST_LIBRARY")
        if not library:
            raise unittest.SkipTest("Set ORT_WEBGPU_TEST_LIBRARY to a native WebGPU plugin")
        cls.dll_directory = os.add_dll_directory(str(Path(library).resolve().parent)) if os.name == "nt" else None
        ort.register_execution_provider_library("WebGpuExecutionProvider", library)
        cls.devices = [d for d in ort.get_ep_devices() if d.ep_name == "WebGpuExecutionProvider"]
        if not cls.devices:
            raise RuntimeError("No WebGPU device")

    def test_gpu_timestamps_and_profiling_lifecycle(self):
        # Reuse the device across sessions, and continue running after EndProfiling.
        graph = helper.make_graph(
            [helper.make_node("Mul", ["x", "x"], ["y"], name="square")],
            "profile",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [65536])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [65536])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
        x = np.linspace(-2, 2, 65536, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            for iteration in range(2):
                options = ort.SessionOptions()
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(directory) / f"profile-{iteration}")
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                options.add_provider_for_devices(
                    self.devices[:1],
                    {
                        "dawnBackendType": os.environ.get("ORT_WEBGPU_TEST_BACKEND", "Vulkan"),
                    },
                )
                session = ort.InferenceSession(model.SerializeToString(), options)
                for _ in range(3):
                    np.testing.assert_array_equal(session.run(None, {"x": x})[0], x * x)
                events = json.loads(Path(session.end_profiling()).read_text())
                shaders = [e for e in events if e.get("cat") == "Api" and "cache_key" in e.get("args", {})]
                self.assertEqual(len(shaders), 3)
                self.assertTrue(all("square" in e["name"] and e["dur"] > 0 and e["ts"] >= 0 for e in shaders))
                self.assertFalse(any(e.get("args", {}).get("provider") == "CPUExecutionProvider" for e in events))
                for _ in range(2):
                    np.testing.assert_array_equal(session.run(None, {"x": x})[0], x * x)
                del session


if __name__ == "__main__":
    unittest.main()
