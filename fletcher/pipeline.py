import argparse

from fletcher.architectures.self_critique import SelfCritique
from fletcher.agents.note_writer import NoteWriter
from fletcher.llm.factory import create_llm_client


def run_pipeline(explanation: str, tone_sample: str, backend: str = "hf") -> str:
    client = create_llm_client(backend)

    critic = SelfCritique(client)
    critique_text = critic.critique(explanation)

    note_writer = NoteWriter(client)
    final_note = note_writer.rewrite(critique_text, tone_sample)

    return final_note


def main():
    parser = argparse.ArgumentParser(description="Run the FLETCHER Architecture 1 pipeline end-to-end.")
    parser.add_argument("--explanation", required=True, help="Student's explanation text.")
    parser.add_argument("--tone-sample", required=True, help="Sample text showing the student's writing tone.")
    parser.add_argument("--backend", default="hf", help="LLM backend to use (hf, fake).")
    args = parser.parse_args()

    result = run_pipeline(args.explanation, args.tone_sample, backend=args.backend)
    print(result)


if __name__ == "__main__":
    main()