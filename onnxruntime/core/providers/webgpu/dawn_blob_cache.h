// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include "core/common/inlined_containers.h"

#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#include <process.h>
#include <share.h>
#else
#include <unistd.h>
#endif

namespace onnxruntime::webgpu {

// A best-effort cache: invalid, missing or unwritable disk entries are cache misses.
// Full keys and checksums are verified before opaque driver blobs reach Dawn.
class DawnBlobCache {
 public:
  explicit DawnBlobCache(std::filesystem::path directory,
                         size_t memory_limit = 64 * 1024 * 1024,
                         size_t disk_limit = 256 * 1024 * 1024)
      : directory_(std::move(directory) / "v1"), memory_limit_(memory_limit), disk_limit_(disk_limit) {
    std::error_code error;
    std::filesystem::create_directories(directory_, error);
    disk_available_ = !error;
    if (disk_available_) TrimDisk();
  }

  size_t FindKey(std::span<const std::byte> key) noexcept {
    try {
      std::lock_guard lock(mutex_);
      auto entry = Find(key);
      return entry ? entry->data.size() : 0;
    } catch (...) {
      return 0;
    }
  }

  size_t LoadData(std::span<const std::byte> key, std::span<std::byte> destination) noexcept {
    try {
      std::lock_guard lock(mutex_);
      auto entry = Find(key);
      if (!entry || destination.size() < entry->data.size()) return 0;
      std::memcpy(destination.data(), entry->data.data(), entry->data.size());
      return entry->data.size();
    } catch (...) {
      return 0;
    }
  }

  void StoreData(std::span<const std::byte> key, std::span<const std::byte> data) noexcept {
    try {
      if (key.empty() || key.size() > kMaxKeySize || data.empty() || data.size() > kMaxEntrySize) return;
      std::lock_guard lock(mutex_);
      const auto token = Token(key);
      auto entry = std::make_shared<Entry>();
      entry->key.assign(key.begin(), key.end());
      entry->data.assign(data.begin(), data.end());
      Remember(token, entry);
      if (disk_available_ && Write(token, *entry)) {
        disk_bytes_ += kHeaderSize + key.size() + data.size();
        if (disk_bytes_ > disk_limit_) TrimDisk();
      }
    } catch (...) {
      // Cache storage failure must never prevent pipeline compilation.
    }
  }

 private:
  static constexpr size_t kMaxKeySize = 1024 * 1024;
  static constexpr size_t kMaxEntrySize = 16 * 1024 * 1024;
  static constexpr size_t kHeaderSize = 32;
  struct Entry {
    InlinedVector<std::byte> key;
    InlinedVector<std::byte> data;
    uint64_t touched = 0;
    size_t Size() const { return key.size() + data.size(); }
  };

  static uint64_t Hash(std::span<const std::byte> data, uint64_t value = 14695981039346656037ull) {
    for (auto byte : data) {
      value ^= std::to_integer<uint8_t>(byte);
      value *= 1099511628211ull;
    }
    return value;
  }

  static std::string Token(std::span<const std::byte> key) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result(32, '0');
    for (size_t half = 0; half != 2; ++half) {
      uint64_t value = Hash(key, half ? 7809847782465536322ull : 14695981039346656037ull);
      for (size_t i = 0; i != 16; ++i) {
        result[half * 16 + i] = hex[value & 15];
        value >>= 4;
      }
    }
    return result;
  }

  static void Put64(std::byte* output, uint64_t value) {
    for (size_t i = 0; i != 8; ++i) output[i] = std::byte((value >> (i * 8)) & 255);
  }
  static uint64_t Get64(const std::byte* input) {
    uint64_t value = 0;
    for (size_t i = 0; i != 8; ++i) value |= uint64_t(std::to_integer<uint8_t>(input[i])) << (i * 8);
    return value;
  }

  std::shared_ptr<Entry> Find(std::span<const std::byte> key) {
    if (key.empty() || key.size() > kMaxKeySize) return nullptr;
    const auto token = Token(key);
    auto cached = entries_.find(token);
    if (cached != entries_.end() && std::ranges::equal(cached->second->key, key)) {
      cached->second->touched = ++tick_;
      return cached->second;
    }
    if (!disk_available_) return nullptr;
    const auto path = directory_ / (token + ".blob");
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size < kHeaderSize || size > kHeaderSize + kMaxKeySize + kMaxEntrySize) return nullptr;
    std::ifstream stream(path, std::ios::binary);
    std::array<std::byte, kHeaderSize> header{};
    if (!stream.read(reinterpret_cast<char*>(header.data()), header.size())) return nullptr;
    if (std::memcmp(header.data(), "ORTDWC01", 8) != 0) return nullptr;
    const auto key_size = Get64(header.data() + 8);
    const auto data_size = Get64(header.data() + 16);
    if (key_size != key.size() || data_size == 0 || data_size > kMaxEntrySize ||
        size != kHeaderSize + key_size + data_size) return nullptr;
    auto entry = std::make_shared<Entry>();
    entry->key.resize(static_cast<size_t>(key_size));
    entry->data.resize(static_cast<size_t>(data_size));
    if (!stream.read(reinterpret_cast<char*>(entry->key.data()), entry->key.size()) ||
        !std::ranges::equal(entry->key, key) ||
        !stream.read(reinterpret_cast<char*>(entry->data.data()), entry->data.size()) ||
        Hash(entry->data, Hash(entry->key)) != Get64(header.data() + 24)) return nullptr;
    Remember(token, entry);
    return entry;
  }

  void Remember(const std::string& token, const std::shared_ptr<Entry>& entry) {
    if (auto old = entries_.find(token); old != entries_.end()) {
      memory_bytes_ -= old->second->Size();
      entries_.erase(old);
    }
    if (entry->Size() > memory_limit_) return;
    while (!entries_.empty() && memory_bytes_ + entry->Size() > memory_limit_) {
      auto oldest = std::min_element(entries_.begin(), entries_.end(), [](const auto& a, const auto& b) {
        return a.second->touched < b.second->touched;
      });
      memory_bytes_ -= oldest->second->Size();
      entries_.erase(oldest);
    }
    entry->touched = ++tick_;
    entries_[token] = entry;
    memory_bytes_ += entry->Size();
  }

  bool Write(const std::string& token, const Entry& entry) {
    static std::atomic<uint64_t> serial{0};
#ifdef _WIN32
    const auto pid = _getpid();
#else
    const auto pid = getpid();
#endif
    const auto temporary = directory_ / (token + ".tmp-" + std::to_string(pid) + "-" + std::to_string(++serial));
#ifdef _WIN32
    int descriptor = -1;
    if (_wsopen_s(&descriptor, temporary.c_str(), _O_CREAT | _O_EXCL | _O_WRONLY | _O_BINARY,
                  _SH_DENYRW, _S_IREAD | _S_IWRITE) != 0) return false;
#else
    int descriptor = open(temporary.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, S_IRUSR | S_IWUSR);
#endif
    if (descriptor < 0) return false;
    auto write = [descriptor](std::span<const std::byte> bytes) {
      while (!bytes.empty()) {
#ifdef _WIN32
        const auto count = _write(descriptor, bytes.data(), static_cast<unsigned>(bytes.size()));
#else
        const auto count = ::write(descriptor, bytes.data(), bytes.size());
#endif
        if (count <= 0) return false;
        bytes = bytes.subspan(static_cast<size_t>(count));
      }
      return true;
    };
    std::array<std::byte, kHeaderSize> header{};
    std::memcpy(header.data(), "ORTDWC01", 8);
    Put64(header.data() + 8, entry.key.size());
    Put64(header.data() + 16, entry.data.size());
    Put64(header.data() + 24, Hash(entry.data, Hash(entry.key)));
    const bool written = write(header) && write(entry.key) && write(entry.data);
#ifdef _WIN32
    const bool closed = _close(descriptor) == 0;
#else
    const bool closed = close(descriptor) == 0;
#endif
    std::error_code error;
    if (written && closed) {
      // Publish complete entries atomically; readers never see a partially written blob.
      std::filesystem::rename(temporary, directory_ / (token + ".blob"), error);
      if (!error) return true;
    }
    std::filesystem::remove(temporary, error);
    return false;
  }

  void TrimDisk() {
    struct File {
      std::filesystem::path path;
      uintmax_t size;
      std::filesystem::file_time_type time;
    };
    InlinedVector<File> files;
    std::error_code error;
    disk_bytes_ = 0;
    for (std::filesystem::directory_iterator it(directory_, error), end; !error && it != end; it.increment(error)) {
      const auto name = it->path().filename().string();
      if (name.size() != 37 || name.substr(32) != ".blob" ||
          !std::all_of(name.begin(), name.begin() + 32, [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
          }) ||
          it->is_symlink(error) || !it->is_regular_file(error)) continue;
      const auto size = it->file_size(error);
      if (error) break;
      const auto time = it->last_write_time(error);
      if (error) break;
      disk_bytes_ += size;
      files.push_back({it->path(), size, time});
    }
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) { return a.time < b.time; });
    for (const auto& file : files) {
      if (disk_bytes_ <= disk_limit_) break;
      if (std::filesystem::remove(file.path, error)) disk_bytes_ -= file.size;
    }
  }

  std::filesystem::path directory_;
  size_t memory_limit_;
  size_t disk_limit_;
  size_t memory_bytes_ = 0;
  uintmax_t disk_bytes_ = 0;
  uint64_t tick_ = 0;
  bool disk_available_ = false;
  std::mutex mutex_;
  InlinedHashMap<std::string, std::shared_ptr<Entry>> entries_;
};

}  // namespace onnxruntime::webgpu
