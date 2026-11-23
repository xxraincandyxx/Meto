import Foundation
import Metal

// Each lesson owns its shapes, a CPU oracle, and the launch geometry of its shaders.
// The harness handles compilation, timing, and comparison in the same way for all.
struct Example {
    let inputs: [[Float]]
    let expected: [Float]
}

struct Lesson {
    let folder: String
    let title: String
    let cases: [[Int]]
    let benchmark: [Int]
    let example: ([Int]) -> Example
    let plan: (Program, Bool, [MTLBuffer], MTLBuffer, [Int]) throws -> [Pass]

    var id: String { String(folder.prefix(2)) }
}

// Reproducible signed inputs expose swapped operands, missing tails and indexing
// mistakes. Binary fractions also make exact checks possible in several lessons.
func samples(_ count: Int, seed: UInt32 = 1) -> [Float] {
    var state = seed
    return (0..<count).map { _ in
        state = state &* 1664525 &+ 1013904223
        return Float(Int(state >> 24) - 128) / 64
    }
}

func linearPass(_ name: String, _ buffers: [MTLBuffer], _ dimensions: [Int],
                count: Int, threads: Int = 256) -> Pass {
    Pass(name, buffers, dimensions, groups: size(ceilDiv(count, threads)), threads: size(threads))
}
