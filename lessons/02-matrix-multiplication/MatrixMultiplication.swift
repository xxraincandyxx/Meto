let matrixMultiplication = Lesson(folder: "02-matrix-multiplication", title: "Matrix multiplication",
    cases: [[1, 1, 1], [2, 3, 4], [33, 17, 37], [64, 32, 64]], benchmark: [384, 384, 384],
    example: { s in
        let (m, k, n) = (s[0], s[1], s[2])
        let a = samples(m*k), b = samples(k*n, seed: 42)
        var c = [Float](repeating: 0, count: m*n)
        for row in 0..<m {
            for inner in 0..<k {
                for col in 0..<n { c[row*n + col] += a[row*k + inner] * b[inner*n + col] }
            }
        }
        return Example(inputs: [a,b], expected: c)
    }, plan: { _, optimized, inputs, output, shape in
        let tile = optimized ? 32 : 16
        return [Pass("multiply", inputs + [output], shape,
                     groups: size(ceilDiv(shape[2], tile), ceilDiv(shape[0], tile)),
                     threads: optimized ? size(8, 8) : size(16, 16))]
    })
