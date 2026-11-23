let reduction = Lesson(folder: "04-reduction", title: "Sum reduction",
    cases: [[1], [17], [256], [1025], [1_050_011]], benchmark: [4_194_304],
    example: { shape in
        let a = samples(shape[0])
        return Example(inputs: [a], expected: [Float(a.reduce(0.0) { $0 + Double($1) })])
    }, plan: { program, optimized, inputs, output, shape in
        // A dispatch can synchronize within a group. The next dispatch combines
        // results across groups, repeating until only one partial sum remains.
        var n = shape[0], source = inputs[0], passes: [Pass] = []
        repeat {
            let groups = ceilDiv(n, optimized ? 1024 : 256)
            let destination = groups == 1 ? output : try program.gpu.buffer(count: groups)
            passes.append(Pass("reduceSum", [source, destination], [n],
                               groups: size(groups), threads: size(256)))
            n = groups
            source = destination
        } while n > 1
        return passes
    })
