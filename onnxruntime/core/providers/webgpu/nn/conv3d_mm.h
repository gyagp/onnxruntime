// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/nn/activation_util.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime::webgpu {

// Implicit im2col readers feed the shared tiled MatMul implementation. No
// expanded input tensor is allocated, even for large 3D filters.
class Conv3dMMProgram final : public Program<Conv3dMMProgram> {
 public:
  Conv3dMMProgram(const Activation& activation, bool channels_last, bool has_bias, bool vec4)
      : Program{"Conv3dMM"}, activation_(activation), channels_last_(channels_last), has_bias_(has_bias), vec4_(vec4) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"dim_a_outer", ProgramUniformVariableDataType::Uint32},
      {"dim_b_outer", ProgramUniformVariableDataType::Uint32},
      {"dim_inner", ProgramUniformVariableDataType::Uint32},
      {"input_spatial", ProgramUniformVariableDataType::Uint32},
      {"output_spatial", ProgramUniformVariableDataType::Uint32},
      {"kernel_spatial", ProgramUniformVariableDataType::Uint32},
      {"input_channels", ProgramUniformVariableDataType::Uint32},
      {"output_channels", ProgramUniformVariableDataType::Uint32},
      {"pads", ProgramUniformVariableDataType::Uint32},
      {"strides", ProgramUniformVariableDataType::Uint32},
      {"dilations", ProgramUniformVariableDataType::Uint32},
      {"logical_dispatch_x", ProgramUniformVariableDataType::Uint32},
      {"logical_dispatch_y", ProgramUniformVariableDataType::Uint32},
      {"logical_dispatch_z", ProgramUniformVariableDataType::Uint32},
      WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES);

 private:
  const Activation activation_;
  const bool channels_last_;
  const bool has_bias_;
  const bool vec4_;
};

Status ApplyConv3dMMProgram(ComputeContext& context, const Activation& activation,
                            bool channels_last, const Tensor& input, const Tensor& dhwio_weight,
                            const Tensor* bias, Tensor& output,
                            const std::vector<uint32_t>& pads, const std::vector<uint32_t>& strides,
                            const std::vector<uint32_t>& dilations);

}  // namespace onnxruntime::webgpu
