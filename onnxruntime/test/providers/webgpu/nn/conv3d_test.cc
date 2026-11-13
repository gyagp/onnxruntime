// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
namespace {

struct Conv3dCase {
  int64_t cin = 4;
  int64_t cout = 8;
  std::array<int64_t, 3> kernel{3, 3, 3};
  std::vector<int64_t> pads{1, 1, 1, 1, 1, 1};
  std::vector<int64_t> strides{1, 1, 1};
  std::vector<int64_t> dilations{1, 1, 1};
  bool bias = true;
  bool relu = false;
  std::string auto_pad;
};

template <typename T>
void RunConv3d(const Conv3dCase& c, bool channels_last, bool initializer) {
  auto ep = DefaultWebGpuExecutionProvider(channels_last);
  ASSERT_NE(ep, nullptr);
  constexpr int64_t batch = 2;
  const std::array<int64_t, 3> spatial{4, 5, 7};
  std::array<int64_t, 3> out;
  for (size_t i = 0; i < 3; ++i) {
    out[i] = (spatial[i] + c.pads[i] + c.pads[i + 3] -
              (c.kernel[i] - 1) * c.dilations[i] - 1) /
                 c.strides[i] +
             1;
  }
  const int64_t input_pixels = spatial[0] * spatial[1] * spatial[2];
  const int64_t output_pixels = out[0] * out[1] * out[2];
  const int64_t taps = c.kernel[0] * c.kernel[1] * c.kernel[2];
  std::vector<T> x(batch * input_pixels * c.cin);
  std::vector<T> w(c.cout * c.cin * taps);
  std::vector<T> b(c.cout);
  auto input_offset = [&](int64_t n, int64_t ch, int64_t d, int64_t h, int64_t width) {
    const int64_t pixel = (d * spatial[1] + h) * spatial[2] + width;
    return channels_last ? (n * input_pixels + pixel) * c.cin + ch
                         : (n * c.cin + ch) * input_pixels + pixel;
  };
  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t ch = 0; ch < c.cin; ++ch) {
      for (int64_t d = 0; d < spatial[0]; ++d) {
        for (int64_t h = 0; h < spatial[1]; ++h) {
          for (int64_t width = 0; width < spatial[2]; ++width) {
            x[input_offset(n, ch, d, h, width)] = T(float((n + ch * 3 + d * 5 + h * 7 + width) % 17 - 8) / 16);
          }
        }
      }
    }
  }
  for (size_t i = 0; i < w.size(); ++i) {
    w[i] = T(float(static_cast<int>(i % 13) - 6) / 32);
  }
  for (int64_t ch = 0; ch < c.cout; ++ch) {
    b[ch] = T(float(ch - 3) / 8);
  }
  std::vector<T> y(batch * output_pixels * c.cout);
  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t ch = 0; ch < c.cout; ++ch) {
      for (int64_t d = 0; d < out[0]; ++d) {
        for (int64_t h = 0; h < out[1]; ++h) {
          for (int64_t width = 0; width < out[2]; ++width) {
            float sum = 0;
            for (int64_t ic = 0; ic < c.cin; ++ic) {
              for (int64_t kd = 0; kd < c.kernel[0]; ++kd) {
                for (int64_t kh = 0; kh < c.kernel[1]; ++kh) {
                  for (int64_t kw = 0; kw < c.kernel[2]; ++kw) {
                    const int64_t id = d * c.strides[0] + kd * c.dilations[0] - c.pads[0];
                    const int64_t ih = h * c.strides[1] + kh * c.dilations[1] - c.pads[1];
                    const int64_t iw = width * c.strides[2] + kw * c.dilations[2] - c.pads[2];
                    if (id >= 0 && id < spatial[0] && ih >= 0 && ih < spatial[1] && iw >= 0 && iw < spatial[2]) {
                      const int64_t wi = ((ch * c.cin + ic) * c.kernel[0] + kd) * c.kernel[1] * c.kernel[2] + kh * c.kernel[2] + kw;
                      sum += float(x[input_offset(n, ic, id, ih, iw)]) * float(w[wi]);
                    }
                  }
                }
              }
            }
            // Bias/activation are applied after the MatMul result has been converted to T.
            T value(sum);
            if (c.bias) value = T(float(value) + float(b[ch]));
            if (c.relu) value = T(std::max(0.0f, float(value)));
            const int64_t pixel = (d * out[1] + h) * out[2] + width;
            y[channels_last ? (n * output_pixels + pixel) * c.cout + ch
                            : (n * c.cout + ch) * output_pixels + pixel] = value;
          }
        }
      }
    }
  }
  OpTester test("Conv", 11, channels_last ? kMSInternalNHWCDomain : kOnnxDomain);
  test.AddAttribute("kernel_shape", std::vector<int64_t>(c.kernel.begin(), c.kernel.end()));
  if (c.auto_pad.empty()) {
    test.AddAttribute("pads", c.pads);
  } else {
    test.AddAttribute("auto_pad", c.auto_pad);
  }
  test.AddAttribute("strides", c.strides);
  test.AddAttribute("dilations", c.dilations);
  if (c.relu) test.AddAttribute("activation", std::string("Relu"));
  test.AddInput<T>("X", channels_last ? std::vector<int64_t>{batch, spatial[0], spatial[1], spatial[2], c.cin} : std::vector<int64_t>{batch, c.cin, spatial[0], spatial[1], spatial[2]}, x);
  test.AddInput<T>("W", {c.cout, c.cin, c.kernel[0], c.kernel[1], c.kernel[2]}, w, initializer);
  if (c.bias)
    test.AddInput<T>("B", {c.cout}, b);
  else
    test.AddOptionalInputEdge<T>();
  test.AddOutput<T>("Y", channels_last ? std::vector<int64_t>{batch, out[0], out[1], out[2], c.cout} : std::vector<int64_t>{batch, c.cout, out[0], out[1], out[2]}, y, false, 0.001f, 0.001f);
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(options).ConfigEp(std::move(ep)).RunWithConfig();
}

template <typename T>
void RunLayouts(const Conv3dCase& c) {
  for (bool layout : {false, true}) {
    for (bool initializer : {false, true}) {
      SCOPED_TRACE(layout);
      SCOPED_TRACE(initializer);
      RunConv3d<T>(c, layout, initializer);
    }
  }
}

}  // namespace

TEST(Conv3dWebGpuTest, Spatial) { RunLayouts<float>({}); }
TEST(Conv3dWebGpuTest, SpatialHalf) { RunLayouts<MLFloat16>({}); }
TEST(Conv3dWebGpuTest, ScalarTail) {
  Conv3dCase c;
  c.cin = 3;
  c.cout = 5;
  c.bias = false;
  RunLayouts<float>(c);
}
TEST(Conv3dWebGpuTest, StrideDilationAndAsymmetricPads) {
  Conv3dCase c;
  c.kernel = {2, 3, 2};
  c.pads = {1, 0, 2, 0, 2, 1};
  c.strides = {2, 1, 3};
  c.dilations = {1, 2, 1};
  RunLayouts<float>(c);
}
TEST(Conv3dWebGpuTest, Pointwise) {
  Conv3dCase c;
  c.kernel = {1, 1, 1};
  c.pads = {0, 0, 0, 0, 0, 0};
  RunLayouts<float>(c);
  RunLayouts<MLFloat16>(c);
  c.cin = 16;
  c.cout = 97;
  RunLayouts<float>(c);
  RunLayouts<MLFloat16>(c);
}
TEST(Conv3dWebGpuTest, PointwiseWithPaddingAndStride) {
  Conv3dCase c;
  c.kernel = {1, 1, 1};
  c.pads = {0, 0, 0, 1, 2, 1};
  RunLayouts<float>(c);
  c.pads = {0, 0, 0, 0, 0, 0};
  c.strides = {1, 2, 3};
  RunLayouts<float>(c);
}
TEST(Conv3dWebGpuTest, AutoPadding) {
  Conv3dCase c;
  c.kernel = {2, 3, 2};
  c.strides = {2, 1, 3};
  c.auto_pad = "SAME_UPPER";
  c.pads = {0, 1, 0, 0, 1, 1};
  RunLayouts<float>(c);
  c.auto_pad = "SAME_LOWER";
  c.pads = {0, 1, 1, 0, 1, 0};
  RunLayouts<MLFloat16>(c);
}
TEST(Conv3dWebGpuTest, FusedRelu) {
  Conv3dCase c;
  c.relu = true;
  RunConv3d<float>(c, true, true);
  RunConv3d<MLFloat16>(c, true, false);
}

}  // namespace onnxruntime::test
