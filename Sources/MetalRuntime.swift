import Foundation
import Metal

struct LabError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

func ceilDiv(_ value: Int, _ divisor: Int) -> Int { (value + divisor - 1) / divisor }

func size(_ x: Int, _ y: Int = 1) -> MTLSize {
    MTLSize(width: x, height: y, depth: 1)
}

// A pass describes one grid of threadgroups. Each group contains the same number
// of cooperating threads. Launching full groups lets tiled kernels use barriers
// even when an input ends in a partial tile.
struct Pass {
    let name: String
    let buffers: [MTLBuffer]
    let dimensions: [UInt32]
    let groups: MTLSize
    let threads: MTLSize

    init(_ name: String, _ buffers: [MTLBuffer], _ dimensions: [Int],
         groups: MTLSize, threads: MTLSize) {
        self.name = name
        self.buffers = buffers
        self.dimensions = dimensions.map { UInt32($0) }
        self.groups = groups
        self.threads = threads
    }
}

final class GPU {
    let device: MTLDevice
    let queue: MTLCommandQueue

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw LabError("No Metal compute device is available to this process. Run locally in a graphical macOS session.")
        }
        self.device = device
        self.queue = queue
    }

    func buffer(count: Int) throws -> MTLBuffer {
        // Shared storage gives Swift and Metal access to the same allocation.
        // The CPU must still wait for the GPU to finish writing before reading it.
        guard let buffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride,
                                             options: .storageModeShared) else {
            throw LabError("Could not allocate a buffer for \(count) floats.")
        }
        return buffer
    }

    func upload(_ values: [Float]) throws -> MTLBuffer {
        let result = try buffer(count: values.count)
        values.withUnsafeBytes { bytes in
            result.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        return result
    }
}

final class Program {
    let gpu: GPU
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]

    init(_ gpu: GPU, path: String) throws {
        self.gpu = gpu
        let source = try String(contentsOfFile: path, encoding: .utf8)
        library = try gpu.device.makeLibrary(source: source, options: nil)
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        if let cached = pipelines[name] { return cached }
        guard let function = library.makeFunction(name: name) else {
            throw LabError("Missing Metal kernel '\(name)'. Check the selected shader.")
        }
        let state = try gpu.device.makeComputePipelineState(function: function)
        pipelines[name] = state
        return state
    }

    func run(_ passes: [Pass]) throws -> Double {
        guard let command = gpu.queue.makeCommandBuffer() else { throw LabError("Cannot create command buffer.") }
        for pass in passes {
            let state = try pipeline(pass.name)
            guard pass.threads.width * pass.threads.height <= state.maxTotalThreadsPerThreadgroup else {
                throw LabError("\(pass.name) exceeds the threadgroup limit on \(gpu.device.name).")
            }
            // A separate encoder per dependent pass gives Metal a synchronization
            // boundary for the shared, hazard-tracked buffers (e.g. partial sums).
            guard let encoder = command.makeComputeCommandEncoder() else { throw LabError("Cannot create encoder.") }
            encoder.label = pass.name
            encoder.setComputePipelineState(state)
            for (slot, buffer) in pass.buffers.enumerated() {
                encoder.setBuffer(buffer, offset: 0, index: slot)
            }
            // Scalars follow the buffers in the Metal function signature. Sending
            // their bytes inline avoids allocating a new GPU buffer for each size.
            for (offset, value) in pass.dimensions.enumerated() {
                var scalar = value
                encoder.setBytes(&scalar, length: MemoryLayout<UInt32>.size, index: pass.buffers.count + offset)
            }
            encoder.dispatchThreadgroups(pass.groups, threadsPerThreadgroup: pass.threads)
            encoder.endEncoding()
        }
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw LabError(command.error?.localizedDescription ?? "GPU command failed.")
        }
        return (command.gpuEndTime - command.gpuStartTime) * 1_000
    }
}
