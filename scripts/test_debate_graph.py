from fletcher.architectures.debate import build_debate_graph

app = build_debate_graph()

result = app.invoke({
    "explanation": "A hash table stores values by converting keys into array indices using a hash function.",
    "conceptual_result": "",
    "procedural_result": "",
    "round": 0,
    "final_result": "",
})

print("final state:", result)
