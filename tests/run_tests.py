"""Build both programs and check array ownership, math, and console input."""
import math
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ["-std=c++11", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
         "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-fno-pie", "-no-pie"]


def run(args, cwd, **kwargs):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"{args}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    if result.stderr:
        raise RuntimeError(result.stderr)
    return result


with tempfile.TemporaryDirectory(prefix="neural-network-tests-") as directory:
    work = Path(directory)
    for name, training in (("NN_main", True), ("NN_training", False)):
        binary = work / (name + "_test")
        defines = [f'-DNETWORK_SOURCE="{ROOT / (name + ".cpp")}"']
        if training:
            defines.append("-DHAS_TRAIN")
        run(["g++", *FLAGS, *defines, str(ROOT / "tests/network_test.cpp"), "-o", str(binary)], work)
        run([str(binary)], work)
        run(["g++", *FLAGS, str(ROOT / (name + ".cpp")), "-o", str(work / name)], work)
        print(f"PASS {name}: build, forward pass, ownership" + (", training and weight roundtrip" if training else ""))

    (work / "weights.txt").write_text("0.2 " * 38)
    predictor = str(work / "NN_training")
    result = run([predictor], work, input="2\n1 0 1 0\n0 1 0 1\n")
    predictions = [float(line.split("Предсказание: ")[1])
                   for line in result.stdout.splitlines() if "Предсказание: " in line]
    sigmoid = lambda x: 1 / (1 + math.exp(-x))
    expected = sigmoid(0.4 * sigmoid(1.2 * sigmoid(0.4)))
    assert len(predictions) == 2
    assert all(abs(value - expected) < 1e-6 for value in predictions)
    run([predictor], work, input="0\n")
    for bad_input in ("-1\n", "abc\n", "1\n1 0 abc 0\n", "1\n1 0\n"):
        result = subprocess.run([predictor], cwd=work, input=bad_input, text=True, capture_output=True)
        assert result.returncode == 1, result
        assert "Некорректное" in result.stderr, result.stderr
        assert "Sanitizer" not in result.stderr, result.stderr
    print("PASS console: multiple inputs, zero inputs, invalid count, invalid/incomplete values")
