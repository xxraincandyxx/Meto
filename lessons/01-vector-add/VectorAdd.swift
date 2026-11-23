let vectorAdd = Lesson(folder: "01-vector-add", title: "Vector addition",
    cases: [[1], [3], [4], [257], [1027]], benchmark: [4_194_307],
    example: { shape in
        let a = samples(shape[0]), b = samples(shape[0], seed: 2)
        return Example(inputs: [a, b], expected: zip(a, b).map(+))
    }, plan: { _, optimized, inputs, output, shape in
        let count = optimized ? ceilDiv(shape[0], 4) : shape[0]
        return [linearPass("add", inputs + [output], shape, count: count)]
    })
