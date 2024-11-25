// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { WebGpuBackend } from '../backend-webgpu';
import { LOG_DEBUG } from '../log';

import { GpuData, GpuDataId, GpuDataType } from './types';

/**
 * manages GpuDataId -> GpuBuffer
 */
export interface GpuDataManager {
  /**
   * copy data from CPU to GPU.
   */
  upload(id: GpuDataId, data: Uint8Array): void;
  /**
   * copy data from GPU to GPU.
   */
  memcpy(sourceId: GpuDataId, destinationId: GpuDataId): void;
  /**
   * create new data on GPU.
   */
  getBuffer(descriptor: GPUBufferDescriptor): GpuData;
  /**
   * get GPU data by ID.
   */
  get(id: GpuDataId): GpuData | undefined;
  /**
   * release the data on GPU by ID.
   *
   * @return size of the data released
   */
  release(id: GpuDataId, pending?: boolean): number;
  /**
   * copy data from GPU to CPU.
   */
  download(id: GpuDataId, getTargetBuffer: () => Uint8Array): Promise<void>;

  /**
   * refresh the buffers that marked for release.
   *
   * when release() is called, the buffer is not released immediately. this is because we need to wait for the commands
   * to be submitted to the GPU. this function is called after the commands are submitted so that the buffers can be
   * actually released.
   */
  refreshPendingBuffers(): void;

  /**
   * register an external buffer for IO Binding. If the buffer is already registered, return the existing GPU data ID.
   *
   * GPU data manager only manages a mapping between the buffer and the GPU data ID. It will not manage the lifecycle of
   * the external buffer.
   */
  registerExternalBuffer(buffer: GPUBuffer, originalSize: number, previous?: [GpuDataId, GPUBuffer]): number;

  /**
   * unregister an external buffer for IO Binding.
   */
  unregisterExternalBuffer(id: GpuDataId): void;

  /**
   * destroy all gpu buffers.
   */
  dispose(): void;

  /**
   * release session related data.
   * @param sessionId - specify the session ID.
   */
  onReleaseSession(sessionId: number): void;

  downloadGpuData(gpuBuffer: GPUBuffer, originalSize: number, getTargetBuffer?: () => Uint8Array): Promise<Uint8Array>;
}

interface StorageCacheValue {
  gpuData: GpuData;
  originalSize: number;
}

const bucketFreelist: Map<number, number> = new Map([
  [64, 250],
  [128, 200],
  [256, 200],
  [512, 200],
  [1024, 200],
  [2048, 230],
  [4096, 200],
  [8192, 50],
  [16384, 50],
  [32768, 50],
  [65536, 50],
  [131072, 50],
  [262144, 50],
  [524288, 50],
  [1048576, 50],
  [2097152, 30],
  [4194304, 20],
  [8388608, 10],
  [12582912, 10],
  [16777216, 10],
  [26214400, 15],
  [33554432, 22],
  [44236800, 2],
  [58982400, 6],
  // we don't want to cache the bucket sizes below but not caching them
  // results in some major performance hits for models like sd-turbo.
  [67108864, 6],
  [134217728, 6],
  [167772160, 6],
]);

const bucketArr: number[] = [];

/**
 * normalize the buffer size so that it fits the 128-bits (16 bytes) alignment.
 */
const calcNormalizedBufferSize = (size: number) => Math.ceil(size / 16) * 16;

/**
 * calculate the buffer size so that it fits into buckets.
 */
const calcBucketBufferSize = (size: number) => {
  for (let idx = 0; idx < bucketArr.length; idx++) {
    const sizeForBucket = bucketArr[idx];
    if (size <= sizeForBucket) {
      return sizeForBucket;
    }
  }
  // not in bucket list -> caller will not cache, round up to 16.
  return Math.ceil(size / 16) * 16;
};

let guid = 1;
const createNewGpuDataId = () => guid++;

class GpuDataManagerImpl implements GpuDataManager {
  // GPU Data ID => GPU Data ( storage buffer )
  private storageCache: Map<GpuDataId, StorageCacheValue>;

  // {usage: {size: buffers}}
  private freeBuffers: Map<number, Map<number, GPUBuffer[]>>;

  private pendingBuffers: GPUBuffer[];

  // The pendingBuffers for capture graph.
  // a SessionID -> GPUBuffer[] mapping.
  private capturedPendingBuffers: Map<number, GPUBuffer[]>;

  private totalBufferSize: number;
  private usageBufferSize: Map<number, number>;
  private usageName: Map<number, string>;
  private debug: boolean;
  private pendingUploadCount = 0;

  constructor(private backend: WebGpuBackend) {
    this.storageCache = new Map();
    this.freeBuffers = new Map();
    this.capturedPendingBuffers = new Map();
    this.pendingBuffers = [];

    for (const [key] of bucketFreelist) {
      bucketArr.push(key);
    }
    this.totalBufferSize = 0;
    this.usageBufferSize = new Map();
    this.usageName = new Map();
    this.usageName.set(GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, 'MAP_WRITE|COPY_SRC');
    this.usageName.set(
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      'STORAGE|COPY_SRC|COPY_DST',
    );
    this.usageName.set(GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM, 'COPY_DST|UNIFORM');
    this.debug = this.backend.env.debug!!;
  }

  logBuffer(prefix: string): void {
    let log = `${prefix} totalBufferSize=${this.totalBufferSize}`;
    for (let usage of this.usageBufferSize.keys()) {
      log += `, ${this.usageName.get(usage)}=${this.usageBufferSize.get(usage)!}`;
    }
    LOG_DEBUG('warning', () => log);
  }

  destroyBuffer(buffer: GPUBuffer): void {
    this.totalBufferSize -= buffer.size;
    this.usageBufferSize.set(buffer.usage, this.usageBufferSize.get(buffer.usage)! - buffer.size);
    this.logBuffer(`destroyBuffer: usage=${this.usageName.get(buffer.usage)}, size=${buffer.size}`);
    buffer.destroy();
  }

  /**
   * exported standard download function. This function is used by the session to download the data from GPU, and also by
   * factory to create GPU tensors with the capacity of downloading data from GPU.
   *
   * @param backend - the WebGPU backend
   * @param gpuBuffer - the GPU buffer to download
   * @param originalSize - the original size of the data
   * @param getTargetBuffer - optional. If provided, the data will be copied to the target buffer. Otherwise, a new buffer
   * will be created and returned.
   */
  async downloadGpuData(
    gpuBuffer: GPUBuffer,
    originalSize: number,
    getTargetBuffer?: () => Uint8Array,
  ): Promise<Uint8Array> {
    const bufferSize = calcNormalizedBufferSize(originalSize);
    const gpuStagingBufferData = this.getBuffer({
      size: bufferSize,
      // eslint-disable-next-line no-bitwise
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const gpuStagingBuffer = gpuStagingBufferData.buffer;
    try {
      const commandEncoder = this.backend.getCommandEncoder();
      this.backend.endComputePass();
      commandEncoder.copyBufferToBuffer(
        gpuBuffer /* source buffer */,
        0 /* source offset */,
        gpuStagingBuffer /* destination buffer */,
        0 /* destination offset */,
        bufferSize /* size */,
      );
      this.backend.flush();

      await gpuStagingBuffer.mapAsync(GPUMapMode.READ);

      const arrayBuffer = gpuStagingBuffer.getMappedRange();
      if (getTargetBuffer) {
        // if we already have a CPU buffer to accept the data, no need to clone the ArrayBuffer.
        const targetBuffer = getTargetBuffer();
        targetBuffer.set(new Uint8Array(arrayBuffer, 0, originalSize));
        return targetBuffer;
      } else {
        // the mapped ArrayBuffer will be released when the GPU buffer is destroyed. Need to clone the
        // ArrayBuffer.
        return new Uint8Array(arrayBuffer.slice(0, originalSize));
      }
    } finally {
      this.release(gpuStagingBufferData.id);
    }
  }

  upload(id: GpuDataId, data: Uint8Array): void {
    const srcArrayBuffer = data.buffer;
    const srcOffset = data.byteOffset;
    const srcLength = data.byteLength;
    const size = calcNormalizedBufferSize(srcLength);

    // get destination gpu buffer
    const gpuDataCache = this.storageCache.get(id);
    if (!gpuDataCache) {
      throw new Error('gpu data for uploading does not exist');
    }
    if (gpuDataCache.originalSize !== srcLength) {
      throw new Error(`inconsistent data size. gpu data size=${gpuDataCache.originalSize}, data size=${srcLength}`);
    }

    // eslint-disable-next-line no-bitwise
    const stagingBufferData = this.getBuffer({
      size,
      usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    const stagingBuffer = stagingBufferData.buffer;

    // copy (upload) data
    const arrayBuffer = stagingBuffer.getMappedRange();
    new Uint8Array(arrayBuffer).set(new Uint8Array(srcArrayBuffer, srcOffset, srcLength));
    stagingBuffer.unmap();

    // GPU copy
    const commandEncoder = this.backend.getCommandEncoder();
    this.backend.endComputePass();
    commandEncoder.copyBufferToBuffer(stagingBuffer, 0, gpuDataCache.gpuData.buffer, 0, size);

    this.pendingUploadCount++;
    if (this.pendingUploadCount > 0) {
      this.backend.flush();
      this.pendingUploadCount = 0;
    }

    this.release(stagingBufferData.id, true);

    LOG_DEBUG('verbose', () => `[WebGPU] GpuDataManager.upload(id=${id})`);
  }

  memcpy(sourceId: GpuDataId, destinationId: GpuDataId): void {
    // get source gpu buffer
    const sourceGpuDataCache = this.storageCache.get(sourceId);
    if (!sourceGpuDataCache) {
      throw new Error('source gpu data for memcpy does not exist');
    }
    // get destination gpu buffer
    const destinationGpuDataCache = this.storageCache.get(destinationId);
    if (!destinationGpuDataCache) {
      throw new Error('destination gpu data for memcpy does not exist');
    }
    if (sourceGpuDataCache.originalSize !== destinationGpuDataCache.originalSize) {
      throw new Error('inconsistent source and destination gpu data size');
    }

    const size = calcNormalizedBufferSize(sourceGpuDataCache.originalSize);

    // GPU copy
    const commandEncoder = this.backend.getCommandEncoder();
    this.backend.endComputePass();
    commandEncoder.copyBufferToBuffer(
      sourceGpuDataCache.gpuData.buffer,
      0,
      destinationGpuDataCache.gpuData.buffer,
      0,
      size,
    );
  }

  registerExternalBuffer(buffer: GPUBuffer, originalSize: number, previous?: [GpuDataId, GPUBuffer]): number {
    let id: number | undefined;
    if (previous) {
      id = previous[0];
      if (buffer === previous[1]) {
        LOG_DEBUG(
          'verbose',
          () =>
            `[WebGPU] GpuDataManager.registerExternalBuffer(size=${originalSize}) => id=${
              id
            }, buffer is the same, skip.`,
        );
        return id;
      } else if (this.backend.capturedCommandList.has(this.backend.currentSessionId!)) {
        throw new Error(`Registering a different external buffer under graph capture mode is not supported yet.
             Please use the previous external buffer!`);
      }
    } else {
      id = createNewGpuDataId();
    }

    this.storageCache.set(id, { gpuData: { id, type: GpuDataType.default, buffer }, originalSize });
    LOG_DEBUG(
      'verbose',
      () => `[WebGPU] GpuDataManager.registerExternalBuffer(size=${originalSize}) => id=${id}, registered.`,
    );
    return id;
  }

  unregisterExternalBuffer(id: GpuDataId): void {
    if (id !== undefined) {
      this.storageCache.delete(id);
      LOG_DEBUG('verbose', () => `[WebGPU] GpuDataManager.unregisterExternalBuffer() => id=${id}`);
    }
  }

  // eslint-disable-next-line no-bitwise
  getBuffer(descriptor: GPUBufferDescriptor): GpuData {
    const size = descriptor.size;
    const usage = descriptor.usage;
    const bucketSize = calcBucketBufferSize(size);

    let gpuBuffer;

    if (
      !this.freeBuffers.has(usage) ||
      !this.freeBuffers.get(usage)!.has(bucketSize) ||
      this.freeBuffers.get(usage)!.get(bucketSize)!.length === 0
    ) {
      if (this.debug) {
        this.totalBufferSize += bucketSize;
        if (this.usageBufferSize.has(usage)) {
          this.usageBufferSize.set(usage, this.usageBufferSize.get(usage)! + bucketSize);
        } else {
          this.usageBufferSize.set(usage, bucketSize);
        }
      }

      gpuBuffer = this.backend.device.createBuffer(descriptor);
    } else {
      gpuBuffer = this.freeBuffers.get(usage)!.get(bucketSize)!.pop() as GPUBuffer;
    }

    const gpuData = { id: createNewGpuDataId(), type: GpuDataType.default, buffer: gpuBuffer };
    this.storageCache.set(gpuData.id, { gpuData, originalSize: size });
    if (this.debug) {
      this.logBuffer(
        `[WebGPU] GpuDataManager.getBuffer, usage=${this.usageName.get(usage)}, size=${size}, bucketSize=${bucketSize}, id=${gpuData.id},`,
      );
    }
    return gpuData;
  }

  get(id: GpuDataId): GpuData | undefined {
    return this.storageCache.get(id)?.gpuData;
  }

  release(id: GpuDataId, pending = false): number {
    const cachedData = this.storageCache.get(id);
    if (!cachedData) {
      throw new Error('releasing data does not exist');
    }

    LOG_DEBUG('verbose', () => `[WebGPU] GpuDataManager.release(id=${id}), gpuDataId=${cachedData.gpuData.id}`);

    this.storageCache.delete(id);

    if (pending) {
      this.pendingBuffers.push(cachedData.gpuData.buffer);
    } else {
      const buffer = cachedData.gpuData.buffer;
      const size = buffer.size;
      const usage = buffer.usage;

      LOG_DEBUG('info', () => `[destroyBuffer] bufferId=${id}`);

      if (!this.freeBuffers.has(usage)) {
        this.freeBuffers.set(usage, new Map());
      }
      if (!this.freeBuffers.get(usage)!.has(size)) {
        this.freeBuffers.get(usage)!.set(size, []);
      }

      this.freeBuffers.get(usage)!.get(size)!.push(buffer);
    }

    return cachedData.originalSize;
  }

  async download(id: GpuDataId, getTargetBuffer: () => Uint8Array): Promise<void> {
    const cachedData = this.storageCache.get(id);
    if (!cachedData) {
      throw new Error('data does not exist');
    }
    await this.downloadGpuData(cachedData.gpuData.buffer, cachedData.originalSize, getTargetBuffer);
  }

  refreshPendingBuffers(): void {
    if (this.backend.sessionStatus === 'default') {
      if (this.pendingBuffers.length > 0) {
        this.pendingBuffers.forEach((buffer) => {
          if (buffer.usage & GPUBufferUsage.MAP_READ || buffer.usage & GPUBufferUsage.MAP_WRITE) {
            this.destroyBuffer(buffer);
          }
        });
        this.pendingBuffers = [];
      }
      this.freeBuffers.forEach((usageInfo, usage) => {
        if (!(usage & GPUBufferUsage.STORAGE)) {
          return;
        }
        usageInfo.forEach((buffers, size) => {
          const maxInFreeList = bucketFreelist.get(size);
          const freelist = this.freeBuffers.get(usage)!.get(size) || [];
          if (maxInFreeList === undefined || freelist.length >= maxInFreeList) {
            this.destroyBuffer(buffers[buffers.length - 1]);
          }
        });
      });
    } else {
      // Don't release intermediate tensors in non-default mode.
      // TODO: reuse the storage buffers in non-default mode.
      let capturedBuffers = this.capturedPendingBuffers.get(this.backend.currentSessionId!);
      if (!capturedBuffers) {
        capturedBuffers = [];
        this.capturedPendingBuffers.set(this.backend.currentSessionId!, capturedBuffers);
      }
    }
  }

  dispose() {
    this.freeBuffers.forEach((usageInfo) => {
      usageInfo.forEach((buffers, size) => {
        for (let buffer of buffers) {
          this.destroyBuffer(buffer);
        }
        usageInfo.set(size, []);
      });
    });

    this.storageCache.forEach((storage) => {
      this.destroyBuffer(storage.gpuData.buffer);
    });
    this.storageCache = new Map();

    this.capturedPendingBuffers.forEach((buffers) => {
      buffers.forEach((buffer) => {
        this.destroyBuffer(buffer);
      });
    });
    this.capturedPendingBuffers = new Map();
  }

  onReleaseSession(sessionId: number) {
    // release the captured pending buffers.
    const pendingBuffers = this.capturedPendingBuffers.get(sessionId);
    if (pendingBuffers) {
      pendingBuffers.forEach((buffer) => {
        this.destroyBuffer(buffer);
      });
      this.capturedPendingBuffers.delete(sessionId);
    }
  }
}

export const createGpuDataManager = (...args: ConstructorParameters<typeof GpuDataManagerImpl>): GpuDataManager =>
  new GpuDataManagerImpl(...args);
