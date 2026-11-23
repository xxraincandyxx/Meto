import Foundation

let softmax = Lesson(folder: "05-softmax", title: "Stable softmax",
    cases: [[1], [3], [255], [1025], [8193]], benchmark: [500_000],
    example: { shape in
        // exp(1000) overflows float32. Subtracting the maximum before exp
        // gives the same probabilities while keeping every exponential <= 1.
        let x = samples(shape[0]).map { 1000 + $0 * 3 }
        let peak = x.max()!
        let weights = x.map { exp(Double($0 - peak)) }
        let denominator = weights.reduce(0, +)
        return Example(inputs: [x], expected: weights.map { Float($0 / denominator) })
    }, plan: { program, optimized, inputs, output, shape in
        let n = shape[0]
        if !optimized {
            return [Pass("softmax", inputs + [output], [n], groups: size(1), threads: size(256))]
        }
        let blocks = ceilDiv(n, 1024)
        let stats = try program.gpu.buffer(count: 2 * blocks)
        let global = try program.gpu.buffer(count: 2)
        return [Pass("summarize", inputs + [stats], [n], groups: size(blocks), threads: size(256)),
                Pass("mergeStats", [stats, global], [blocks], groups: size(1), threads: size(256)),
                linearPass("normalize", inputs + [global, output], [n], count: n)]
    })
