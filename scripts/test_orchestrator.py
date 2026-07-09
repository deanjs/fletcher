from fletcher.architectures.debate import orchestrator_node

# case 1: 둘 다 문제 없음
state_ok = {
    "conceptual_verdict": {"role": "conceptual", "flagged": False, "reasoning": "fine", "confidence": 0.9},
    "procedural_verdict": {"role": "procedural", "flagged": False, "reasoning": "fine", "confidence": 0.9},
}
print("case 1 (no issues):", orchestrator_node(state_ok))

# case 2: conceptual만 문제 있음
state_issue = {
    "conceptual_verdict": {"role": "conceptual", "flagged": True, "reasoning": "missing collision handling", "confidence": 0.8},
    "procedural_verdict": {"role": "procedural", "flagged": False, "reasoning": "fine", "confidence": 0.9},
}
print("case 2 (conceptual flagged):", orchestrator_node(state_issue))