from fletcher.agents.schemas import CriticVerdict


class Synthesizer:
    """Combines independent role verdicts into one final feedback string.

    This is Stage 2 of the current Architecture 2 implementation: roles have
    different evaluation targets, so synthesis aggregates rather than debates.
    """

    def synthesize(self, verdicts: dict[str, CriticVerdict | dict]) -> dict:
        parts: list[str] = []
        any_flagged = False

        for role, raw_verdict in verdicts.items():
            if not raw_verdict:
                continue
            verdict = (
                raw_verdict
                if isinstance(raw_verdict, CriticVerdict)
                else CriticVerdict(**raw_verdict)
            )
            if verdict.flagged:
                any_flagged = True
                parts.append(f"{role.capitalize()} issue: {verdict.reasoning}")

        if parts:
            final_result = " ".join(parts)
        else:
            final_result = "No issues found. The explanation is accurate and complete."

        return {
            "issue_found": any_flagged,
            "final_result": final_result,
            "flagged_roles": [
                role
                for role, raw_verdict in verdicts.items()
                if raw_verdict
                and (
                    raw_verdict.flagged
                    if isinstance(raw_verdict, CriticVerdict)
                    else CriticVerdict(**raw_verdict).flagged
                )
            ],
        }
