import Foundation

let attention = Lesson(folder: "06-attention", title: "Scaled dot-product attention",
    cases: [[1, 1, 1], [2, 3, 4], [7, 35, 13], [5, 67, 128]], benchmark: [512, 256, 64],
    example: { s in
        let (m, n, d) = (s[0], s[1], s[2])
        let q = samples(m*d), k = samples(n*d, seed: 7), v = samples(n*d, seed: 19)
        var out = [Float](repeating: 0, count: m*d)
        for row in 0..<m {
            var scores = [Double](repeating: 0, count: n)
            for j in 0..<n {
                for f in 0..<d { scores[j] += Double(q[row*d+f]) * Double(k[j*d+f]) }
                scores[j] /= sqrt(Double(d))
            }
            let high = scores.max()!
            let weights = scores.map { exp($0 - high) }, sum = weights.reduce(0, +)
            for f in 0..<d {
                var result = 0.0
                for j in 0..<n { result += weights[j] * Double(v[j*d+f]) }
                out[row*d+f] = Float(result / sum)
            }
        }
        return Example(inputs: [q, k, v], expected: out)
    }, plan: { program, optimized, inputs, output, shape in
        let (m, n, d) = (shape[0], shape[1], shape[2])
        if optimized {
            // Exactly one hardware SIMD group owns a query. Feature strips in
            // the kernel use the reported execution width as their stride.
            let width = try program.pipeline("attend").threadExecutionWidth
            return [Pass("attend", inputs + [output], [n, d], groups: size(m), threads: size(width))]
        }
        let scores = try program.gpu.buffer(count: m*n)
        return [linearPass("score", [inputs[0], inputs[1], scores], shape, count: m*n),
                linearPass("mixValues", [scores, inputs[2], output], shape, count: m*d)]
    })
