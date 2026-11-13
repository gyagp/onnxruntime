// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/conv3d_mm.h"

#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/providers/webgpu/math/gemm_utils.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime::webgpu {

Status Conv3dMMProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& w = shader.AddInput("w", ShaderUsage::UseValueTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);
  const ShaderVariableHelper* bias = has_bias_ ? &shader.AddInput("bias", ShaderUsage::UseValueTypeAlias) : nullptr;
  const uint32_t components = vec4_ ? 4 : 1;
  const std::string value_type = vec4_ ? "vec4<x_element_t>" : "x_element_t";
  auto& code = shader.AdditionalImplementation();
  code << GetActivationDeclaration(activation_, value_type, "x_element_t")
       << "fn read_x(batch: i32, pixel: i32, k: i32) -> x_value_t {\n"
       << "  let in_shape = vec3<i32>(uniforms.input_spatial);\n"
       << "  let out_shape = vec3<i32>(uniforms.output_spatial);\n"
       << "  let kernel_shape = vec3<i32>(uniforms.kernel_spatial);\n"
       << "  let channels = i32(uniforms.input_channels);\n"
       << "  let tap = k / channels;\n"
       << "  let channel = k % channels;\n"
       << "  let out_pos = vec3<i32>(pixel / (out_shape.y * out_shape.z),\n"
       << "                            pixel / out_shape.z % out_shape.y, pixel % out_shape.z);\n"
       << "  let tap_pos = vec3<i32>(tap / (kernel_shape.y * kernel_shape.z),\n"
       << "                            tap / kernel_shape.z % kernel_shape.y, tap % kernel_shape.z);\n"
       << "  let pos = out_pos * vec3<i32>(uniforms.strides) +\n"
       << "            tap_pos * vec3<i32>(uniforms.dilations) - vec3<i32>(uniforms.pads);\n"
       << "  if (any(pos < vec3<i32>(0)) || any(pos >= in_shape)) { return x_value_t(0); }\n"
       << "  let spatial = (pos.x * in_shape.y + pos.y) * in_shape.z + pos.z;\n"
       << "  let batch_size = in_shape.x * in_shape.y * in_shape.z * channels;\n";
  if (channels_last_) {
    code << "  let offset = (batch * batch_size + spatial * channels + channel) / " << components << ";\n";
  } else {
    code << "  let offset = batch * batch_size + channel * in_shape.x * in_shape.y * in_shape.z + spatial;\n";
  }
  code << "  return " << x.GetByOffset("offset") << ";\n}\n"
       << "fn read_w(k: i32, channel: i32) -> w_value_t {\n"
       << "  return " << w.GetByOffset("k * (i32(uniforms.output_channels) / " + std::to_string(components) + ") + channel")
       << ";\n}\n";
  if (channels_last_) {
    code << "fn mm_readA(batch: i32, row: i32, col: i32) -> x_value_t {\n"
         << "  if (row >= i32(uniforms.dim_a_outer) || col * " << components
         << " >= i32(uniforms.dim_inner)) { return x_value_t(0); }\n"
         << "  return read_x(batch, row, col * " << components << ");\n}\n"
         << "fn mm_readB(batch: i32, row: i32, col: i32) -> w_value_t {\n"
         << "  if (row >= i32(uniforms.dim_inner) || col * " << components
         << " >= i32(uniforms.dim_b_outer)) { return w_value_t(0); }\n"
         << "  return read_w(row, col);\n}\n";
  } else {
    code << "fn mm_readA(batch: i32, row: i32, col: i32) -> w_value_t {\n"
         << "  if (row >= i32(uniforms.dim_inner) || col >= i32(uniforms.dim_a_outer)) { return w_value_t(0); }\n"
         << "  return read_w(row, col);\n}\n"
         << "fn mm_readB(batch: i32, row: i32, col: i32) -> x_value_t {\n"
         << "  if (row >= i32(uniforms.dim_inner) || col >= i32(uniforms.dim_b_outer)) { return x_value_t(0); }\n"
         << "  return read_x(batch, col, row);\n}\n";
  }
  code << "fn mm_write(batch: i32, row: i32, col: i32, value_in: " << value_type << ") {\n"
       << "  if (row >= i32(uniforms.dim_a_outer) || col * " << components
       << " >= i32(uniforms.dim_b_outer)) { return; }\n"
       << "  var value = value_in;\n";
  if (has_bias_) {
    code << "  value += " << bias->GetByOffset(channels_last_ ? "col" : "row") << ";\n";
  }
  code << "  " << GetActivationSnippet(activation_, value_type, "x_element_t") << "\n"
       << "  let offset = (batch * i32(uniforms.dim_a_outer) + row) *\n"
       << "               (i32(uniforms.dim_b_outer) / " << components << ") + col;\n"
       << "  " << output.SetByOffset("offset", "value") << "\n}\n";
  const InlinedVector<int64_t> elements_per_thread{4, 4, 1};
  return vec4_ ? MakeMatMulPackedVec4Source(shader, elements_per_thread, 8, 8, "x_element_t", nullptr,
                                            false, false, 1.0f, true, 4, 32, false, 0, true)
               : MakeMatMulPackedSource(shader, elements_per_thread, 8, 8, "x_element_t", nullptr,
                                        !channels_last_, false, 1.0f, true, 32, false, 0, true);
}

Status ApplyConv3dMMProgram(ComputeContext& context, const Activation& activation,
                            bool channels_last, const Tensor& input, const Tensor& dhwio_weight,
                            const Tensor* bias, Tensor& output,
                            const std::vector<uint32_t>& pads, const std::vector<uint32_t>& strides,
                            const std::vector<uint32_t>& dilations) {
  const auto& x_shape = input.Shape();
  const auto& w_shape = dhwio_weight.Shape();
  const auto& y_shape = output.Shape();
  const size_t spatial_start = channels_last ? 1 : 2;
  const uint32_t cin = narrow<uint32_t>(w_shape[3]);
  const uint32_t cout = narrow<uint32_t>(w_shape[4]);
  const uint32_t spatial = SafeInt<uint32_t>(y_shape[spatial_start]) *
                           y_shape[spatial_start + 1] * y_shape[spatial_start + 2];
  const uint32_t k = SafeInt<uint32_t>(cin) * w_shape[0] * w_shape[1] * w_shape[2];
  const bool vec4 = channels_last && cin % 4 == 0 && cout % 4 == 0;
  const int components = vec4 ? 4 : 1;
  const uint32_t m = channels_last ? spatial : cout;
  const uint32_t n = channels_last ? cout : spatial;
  const uint32_t dispatch_x = CeilDiv(n, 32u);
  const uint32_t dispatch_y = CeilDiv(m, 32u);
  const uint32_t batch = narrow<uint32_t>(x_shape[0]);
  const std::vector<uint32_t> input_spatial{narrow<uint32_t>(x_shape[spatial_start]),
                                            narrow<uint32_t>(x_shape[spatial_start + 1]),
                                            narrow<uint32_t>(x_shape[spatial_start + 2])};
  const std::vector<uint32_t> output_spatial{narrow<uint32_t>(y_shape[spatial_start]),
                                             narrow<uint32_t>(y_shape[spatial_start + 1]),
                                             narrow<uint32_t>(y_shape[spatial_start + 2])};
  const std::vector<uint32_t> kernel_spatial{narrow<uint32_t>(w_shape[0]), narrow<uint32_t>(w_shape[1]), narrow<uint32_t>(w_shape[2])};
  const std::vector<uint32_t> head_pads{pads[0], pads[1], pads[2]};
  Conv3dMMProgram program(activation, channels_last, bias != nullptr, vec4);
  program.CacheHint(channels_last, bias != nullptr, vec4, activation.CacheKey())
      .AddInput({&input, ProgramTensorMetadataDependency::TypeAndRank, components})
      .AddInput({&dhwio_weight, ProgramTensorMetadataDependency::TypeAndRank, components})
      .AddOutput({&output, ProgramTensorMetadataDependency::TypeAndRank, components})
      .SetWorkgroupSize(8, 8, 1)
      .SetDispatchGroupSize(dispatch_x, dispatch_y, batch)
      .AddUniformVariables({{m}, {n}, {k}, {input_spatial}, {output_spatial}, {kernel_spatial}, {cin}, {cout}, {head_pads}, {strides}, {dilations}, {dispatch_x}, {dispatch_y}, {batch}});
  AppendActivationUniformsData(activation, program);
  if (bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank, components});
  }
  return context.RunProgram(program);
}

}  // namespace onnxruntime::webgpu
