import argparse
import json
from pathlib import Path

from fletcher.skills import SkillBank, distill_skills_from_debate_log


DEFAULT_INPUT = Path("eval/debate_logs")
DEFAULT_OUTPUT = Path("fletcher/skills/skill_bank.jsonl")


def iter_log_paths(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.glob("*.jsonl") if p.is_file())
    return [path]


def iter_logs(paths: list[Path]):
    for path in paths:
        with path.open() as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    yield path, line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc


def distill_logs(input_path: Path, output_path: Path, reset: bool = False) -> dict:
    paths = iter_log_paths(input_path)
    if not paths:
        raise FileNotFoundError(f"No JSONL debate logs found under {input_path}")

    if reset:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")

    bank = SkillBank(output_path)
    logs_seen = 0
    skills_created = 0

    for _, _, log in iter_logs(paths):
        logs_seen += 1
        skills = distill_skills_from_debate_log(log)
        skills_created += len(skills)
        bank.extend(skills)

    bank.save()
    return {
        "input_files": len(paths),
        "logs_seen": logs_seen,
        "skills_created": skills_created,
        "skills_in_bank": len(bank.skills),
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill debate JSONL logs into a persistent SkillBank.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Debate log JSONL file or directory containing *.jsonl files.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output SkillBank JSONL path.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the output SkillBank before writing distilled skills.",
    )
    args = parser.parse_args()

    summary = distill_logs(Path(args.input), Path(args.output), reset=args.reset)
    print(
        "Distilled "
        f"{summary['skills_created']} skills from {summary['logs_seen']} logs "
        f"across {summary['input_files']} file(s). "
        f"SkillBank now has {summary['skills_in_bank']} skills: {summary['output_path']}"
    )


if __name__ == "__main__":
    main()
