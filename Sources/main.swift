import Foundation
import Metal

let lessons: [Lesson] = [vectorAdd, matrixMultiplication, matrixTranspose, reduction, softmax, attention]
let usage = """
Usage: ./meto list | doctor | check [all|ID] [solution|baseline|practice]
       ./meto bench ID [solution|baseline|practice] | practice ID
IDs follow the course order. For example: ./meto check 01
Bench compares baseline and solution by default. Practice creates an editable baseline.
"""

func check(_ output: MTLBuffer, expected: [Float], id: String) throws {
    let actual = output.contents().assumingMemoryBound(to: Float.self)
    for (index, target) in expected.enumerated() {
        // The first four lessons use dyadic test data and small enough sums to be
        // represented exactly. Probabilities need relative tolerance even near zero.
        let tolerance: Float
        switch id {
        case "05": tolerance = 1e-10 + 3e-5 * abs(target)
        case "06": tolerance = 2e-5 + 3e-5 * abs(target)
        default: tolerance = 0
        }
        guard actual[index].isFinite, abs(actual[index] - target) <= tolerance else {
            throw LabError("output[\(index)] = \(actual[index]), expected \(target) (tolerance \(tolerance))")
        }
    }
}

func exercise(_ lesson: Lesson, gpu: GPU, variant: String, timed: Bool) throws -> Double {
    let path = "lessons/\(lesson.folder)/\(variant).metal"
    guard FileManager.default.fileExists(atPath: path) else {
        throw LabError("Missing \(path). Create a practice file with ./meto practice \(lesson.id).")
    }
    let program = try Program(gpu, path: path)
    var median = 0.0
    for shape in timed ? [lesson.benchmark] : lesson.cases {
        try autoreleasepool {
            let example = lesson.example(shape)
            let inputs = try example.inputs.map(gpu.upload)
            let out = try gpu.buffer(count: example.expected.count)
            // Poison output first: missing writes must fail even if the answer is zero.
            out.contents().assumingMemoryBound(to: Float.self).initialize(repeating: .nan, count: example.expected.count)
            let passes = try lesson.plan(program, variant == "solution", inputs, out, shape)
            _ = try program.run(passes)
            do { try check(out, expected: example.expected, id: lesson.id) }
            catch { throw LabError("\(path) \(shape): \(error)") }
            if timed {
                _ = try program.run(passes)
                var times: [Double] = []
                for _ in 0..<15 { times.append(try program.run(passes)) }
                median = times.sorted()[times.count / 2]
                guard median > 0 else { throw LabError("GPU timestamps unavailable.") }
                print(String(format: "  %@ %@  %.4f ms median GPU time", variant, String(describing: shape), median))
            } else {
                print("  PASS \(variant) \(shape)")
            }
        }
    }
    return median
}

func main() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    let action = args.first ?? "help"
    if ["help", "--help", "-h"].contains(action) { print(usage); return }
    if action == "list", args.count == 1 {
        for lesson in lessons { print("\(lesson.id)  \(lesson.title)") }
        return
    }
    if action == "doctor", args.count == 1 {
        let gpu = try GPU()
        print("Metal device: \(gpu.device.name)")
        _ = try exercise(vectorAdd, gpu: gpu, variant: "solution", timed: false)
        print("Swift host, runtime shader compiler and GPU dispatch are working.")
        return
    }
    guard ["practice", "check", "bench"].contains(action), args.count <= 3 else { throw LabError(usage) }
    let selector = args.count > 1 ? args[1] : "all"
    let selected = lessons.filter { ($0.id == selector || Int($0.id) == Int(selector)) || (["check", "bench"].contains(action) && selector == "all") }
    guard !selected.isEmpty else { throw LabError("Choose a lesson with ./meto list.\n" + usage) }
    if action == "practice" {
        guard args.count == 2 else { throw LabError(usage) }
        let folder = "lessons/" + selected[0].folder
        let path = folder + "/practice.metal"
        if !FileManager.default.fileExists(atPath: path) {
            try FileManager.default.copyItem(atPath: folder + "/baseline.metal", toPath: path)
        }
        print("Edit \(path), then run ./meto check \(selected[0].id) practice")
        return
    }
    let variants = args.count == 3 ? [args[2]] : (action == "bench" ? ["baseline", "solution"] : ["solution"])
    guard variants.allSatisfy({ ["solution", "baseline", "practice"].contains($0) }) else { throw LabError(usage) }
    let gpu = try GPU()
    print(gpu.device.name)
    for lesson in selected {
        print("\(lesson.id)  \(lesson.title)")
        var results: [Double] = []
        for variant in variants { results.append(try exercise(lesson, gpu: gpu, variant: variant, timed: action == "bench")) }
        if results.count == 2 { print(String(format: "  baseline / solution = %.2fx", results[0] / results[1])) }
    }
}

do { try main() } catch {
    FileHandle.standardError.write(Data("Meto: \(error)\n".utf8))
    exit(1)
}
