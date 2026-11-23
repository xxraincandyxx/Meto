"""Offline lesson discovery and practice copies, using the Python standard library."""
import argparse
import ast
from html.parser import HTMLParser
from pathlib import Path
import shutil
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference/leetgpu-challenges/challenges"


def entries():
    paths = REFERENCE.glob("*/*/challenge.py")
    lessons = {int(p.parent.name.split("_")[0]): p for p in paths}
    if not lessons:
        raise RuntimeError("Fetch the reference with git submodule update --init.")
    return dict(sorted(lessons.items()))


class Statement(HTMLParser):
    """Extract the opening problem statement; preserve inline code and skip drawings."""
    def __init__(self):
        super().__init__()
        self.intro = []
        self.done = False

    def handle_starttag(self, tag, attrs):
        if tag in {"h2", "svg"}:
            self.done = True

    def handle_data(self, data):
        if not self.done:
            self.intro.append(data)


def title(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "name" for t in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    return path.parent.name.split("_", 1)[1].replace("_", " ")


def lesson_folder(number, path):
    if number <= 6:
        return next((ROOT / "lessons").glob(f"{number:02}-*"))
    return ROOT / "catalog" / path.parent.name


def practice(folder):
    # The shader helper resolves its .metal sibling from the Python filename.
    # Keep both editable files together and preserve all previous attempts.
    sources = [folder / "baseline.metal"] if (folder / "baseline.metal").exists() else [
        file for file in (folder / "solution.py", folder / "solution.metal") if file.exists()
    ]
    for source in sources:
        target = source.with_name("practice" + source.suffix)
        if not target.exists():
            shutil.copyfile(source, target)
        print(target.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["list", "show", "practice"])
    parser.add_argument("id", type=int, nargs="?")
    args = parser.parse_args()
    lessons = entries()
    if args.action == "list" and args.id is None:
        for number, path in lessons.items():
            print(f"{number:02}  {title(path)}")
        return
    if args.id not in lessons:
        parser.error("Choose a lesson ID from ./metal-lab list.")
    path = lessons[args.id]
    folder = lesson_folder(args.id, path)
    if args.action == "practice":
        practice(folder)
        print(f"Run ./metal-lab check {args.id} practice")
        return
    statement = Statement()
    statement.feed(path.with_name("challenge.html").read_text())
    print(f"{args.id:02}  {title(path)} ({path.parent.parent.name})\n")
    print(textwrap.fill(" ".join("".join(statement.intro).split()), width=86))
    print(f"\nProblem and examples  {path.with_name('challenge.html').relative_to(ROOT)}")
    print(f"Lesson files          {folder.relative_to(ROOT)}")
    print(f"Check the solution    ./metal-lab check {args.id}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        sys.exit(str(exc))
