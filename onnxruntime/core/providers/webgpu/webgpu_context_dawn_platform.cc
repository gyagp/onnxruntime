// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_context_dawn_platform.h"
#include "core/providers/webgpu/dawn_blob_cache.h"
#include "core/common/logging/logging.h"
#include "core/platform/env.h"

#include <algorithm>
#include <memory>
#include <thread>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "dawn/native/DawnNative.h"
#include "dawn/platform/DawnPlatform.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace onnxruntime::webgpu {
namespace {

uint32_t GetDawnWorkerThreadCount() {
  return std::max(2u, std::thread::hardware_concurrency() / 2u);
}

class DawnCachingInterface final : public dawn::platform::CachingInterface {
 public:
  explicit DawnCachingInterface(const std::string& directory)
      : cache_(std::filesystem::path(std::u8string(directory.begin(), directory.end()))) {}
  size_t FindKey(std::span<const std::byte> key) override {
    return cache_.FindKey(key);
  }
  size_t LoadData(std::span<const std::byte> key, std::span<std::byte> destination) override {
    const auto size = cache_.LoadData(key, destination);
    LOGS_DEFAULT(VERBOSE) << "Dawn blob cache " << (size ? "hit" : "miss") << ": " << size << " bytes";
    return size;
  }
  void StoreData(std::span<const std::byte> key, std::span<const std::byte> data) override {
    cache_.StoreData(key, data);
    LOGS_DEFAULT(VERBOSE) << "Dawn blob cache store: " << data.size() << " bytes";
  }

 private:
  DawnBlobCache cache_;
};

class DawnPlatform final : public dawn::platform::Platform {
 public:
  DawnPlatform() {
    const auto directory = Env::Default().GetEnvironmentVar("ORT_WEBGPU_DAWN_CACHE_DIR");
    if (!directory.empty()) {
      try {
        cache_ = std::make_unique<DawnCachingInterface>(directory);
      } catch (...) {
        LOGS_DEFAULT(WARNING) << "Dawn blob cache initialization failed; continuing without caching.";
      }
    }
  }
  dawn::platform::CachingInterface* GetCachingInterface() override { return cache_.get(); }
  std::unique_ptr<dawn::platform::WorkerTaskPool> CreateWorkerTaskPool() override {
    return dawn::platform::WorkerTaskPool::CreateDawnDefault(GetDawnWorkerThreadCount());
  }

 private:
  std::unique_ptr<DawnCachingInterface> cache_;
};

DawnPlatform& GetDawnPlatform() {
  // Dawn retains this non-owning pointer; keep it alive through instance teardown.
  static DawnPlatform* platform = new DawnPlatform();
  return *platform;
}

}  // namespace

const DawnProcTable& GetBundledDawnProcs() {
  return dawn::native::GetProcs();
}

wgpu::Instance CreateBundledDawnInstance(wgpu::InstanceDescriptor instance_desc) {
  dawn::native::DawnInstanceDescriptor dawn_instance_desc{};
  dawn_instance_desc.platform = &GetDawnPlatform();
  instance_desc.nextInChain = &dawn_instance_desc;
  return wgpu::CreateInstance(&instance_desc);
}

InlinedVector<wgpu::Adapter> EnumerateBundledDawnAdapters(WGPUInstance instance,
                                                          const wgpu::RequestAdapterOptions& options) {
  dawn::native::Instance native_instance(reinterpret_cast<dawn::native::InstanceBase*>(instance));
  const auto adapters = native_instance.EnumerateAdapters(&options);
  InlinedVector<wgpu::Adapter> result;
  result.reserve(adapters.size());
  for (const auto& adapter : adapters) {
    result.emplace_back(adapter.Get());
  }
  return result;
}

}  // namespace onnxruntime::webgpu
