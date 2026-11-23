import Foundation

let attention = Lesson(folder: "06-attention", title: "Scaled dot-product attention",
    cases: [[1, 1, 1], [2, 3, 4], [8, 24, 16], [7, 35, 13], [5, 67, 128], [3, 259, 11]], benchmark: [512, 256, 64],
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
            // A short row fits in shared memory. Longer rows stream keys
            // through registers, keeping the scratch space bounded.
            if n <= 256, m % 8 == 0, n % 8 == 0, d % 8 == 0 {
                let width = try program.pipeline("attendFragments").threadExecutionWidth
                return [Pass("attendFragments", inputs + [output], [n, d],
                             groups: size(m/8), threads: size(4*width))]
            }
            let name = n <= 256 ? "attendShort" : "attend"
            let width = try program.pipeline(name).threadExecutionWidth
            let groupsPerQuery = n <= 256 ? 4 : 1
            return [Pass(name, inputs + [output], [n, d], groups: size(m), threads: size(groupsPerQuery*width))]
        }
        let scores = try program.gpu.buffer(count: m*n)
        return [linearPass("score", [inputs[0], inputs[1], scores], shape, count: m*n),
                linearPass("mixValues", [scores, inputs[2], output], shape, count: m*d)]
    })
