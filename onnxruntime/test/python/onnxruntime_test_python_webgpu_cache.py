# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Exercise the native Dawn cache across independent processes and corrupt entries."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHILD = r"""
import json, os, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto
library, cache, backend, operation = sys.argv[1:]
os.environ['ORT_WEBGPU_DAWN_CACHE_DIR'] = cache
dlls = os.add_dll_directory(str(Path(library).parent)) if os.name == 'nt' else None
ort.set_default_logger_severity(0)
ort.set_default_logger_verbosity(1)
ort.register_execution_provider_library('WebGpuExecutionProvider', library)
devices = [d for d in ort.get_ep_devices() if d.ep_name == 'WebGpuExecutionProvider']
assert devices
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.log_severity_level = 3
options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
options.add_provider_for_devices(devices[:1], {'dawnBackendType': backend})
graph = helper.make_graph([helper.make_node(operation, ['x','x'], ['y'])], 'cache',
    [helper.make_tensor_value_info('x', TensorProto.FLOAT, [65536])],
    [helper.make_tensor_value_info('y', TensorProto.FLOAT, [65536])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',21)], ir_version=10)
session = ort.InferenceSession(model.SerializeToString(), options)
x = np.linspace(-2,2,65536,dtype=np.float32)
expected = x*x if operation == 'Mul' else x+x
for _ in range(2):
    np.testing.assert_array_equal(session.run(None,{'x':x})[0], expected)
del session
print(json.dumps({'correct':True}))
"""


class TestWebGpuDawnCache(unittest.TestCase):
    def test_persistence_shader_identity_corruption_and_concurrent_writers(self):
        library = os.environ.get("ORT_WEBGPU_TEST_LIBRARY")
        if not library:
            self.skipTest("Set ORT_WEBGPU_TEST_LIBRARY to the built WebGPU plugin")
        backend = os.environ.get("ORT_WEBGPU_TEST_BACKEND", "Vulkan")

        def run(cache, operation="Mul"):
            process = subprocess.run(
                [sys.executable, "-c", CHILD, str(Path(library).resolve()), str(cache), backend, operation],
                capture_output=True,
                cwd=tempfile.gettempdir(),
                timeout=60,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr[-3000:])
            self.assertTrue(json.loads(process.stdout)["correct"])
            # The Windows native logger writes UTF16LE; Python and POSIX logs use UTF8.
            logs = process.stderr.replace(b"\x00", b"").decode("utf-8", errors="replace")
            return logs

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cold = run(cache)
            self.assertIn("Dawn blob cache store", cold)
            self.assertTrue(list((cache / "v1").glob("*.blob")))
            warm = run(cache)
            self.assertIn("Dawn blob cache hit", warm)
            changed = run(cache, "Add")
            self.assertIn("Dawn blob cache store", changed)
            for file in (cache / "v1").glob("*.blob"):
                with file.open("r+b") as stream:
                    stream.seek(-1, os.SEEK_END)
                    value = stream.read(1)
                    stream.seek(-1, os.SEEK_END)
                    stream.write(bytes([value[0] ^ 255]))
            repaired = run(cache)
            self.assertIn("Dawn blob cache store", repaired)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(run, root / "concurrent") for _ in range(2)]
                for future in futures:
                    future.result()
            self.assertIn("Dawn blob cache hit", run(root / "concurrent"))
            blocked = root / "not-a-directory"
            blocked.write_text("cache unavailable")
            run(blocked)


if __name__ == "__main__":
    unittest.main()
