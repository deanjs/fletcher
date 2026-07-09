from fletcher.architectures.debate import route_after_orchestrator, debate_round_node, MAX_ROUNDS

# case 1: issue found, round has not reached the limit -> should continue debating
state_continue = {"issue_found": True, "round": 0}
print("case 1 (should continue):", route_after_orchestrator(state_continue))

# case 2: issue found, but round limit reached -> should stop
state_limit = {"issue_found": True, "round": MAX_ROUNDS}
print("case 2 (limit reached):", route_after_orchestrator(state_limit))

# case 3: no issue found -> should go straight to synthesizer
state_no_issue = {"issue_found": False, "round": 0}
print("case 3 (no issue):", route_after_orchestrator(state_no_issue))

# check that debate_round_node increments the counter
print("round increment:", debate_round_node({"round": 0}))