let matrixTranspose = Lesson(folder: "03-matrix-transpose", title: "Matrix transpose",
    cases: [[1, 1], [3, 1], [1, 35], [33, 65]], benchmark: [2049, 2051],
    example: { s in
        let rows = s[0], cols = s[1], a = samples(rows * cols)
        return Example(inputs: [a], expected: (0..<rows*cols).map { a[($0 % rows)*cols + $0/rows] })
    }, plan: { _, optimized, inputs, output, shape in
        let tile = optimized ? 32 : 16
        return [Pass("transpose", inputs + [output], shape,
                     groups: size(ceilDiv(shape[1], tile), ceilDiv(shape[0], tile)),
                     threads: optimized ? size(8, 8) : size(16, 16))]
    })
