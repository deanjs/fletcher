# FLETCHER Study Roadmap

이 문서는 FLETCHER를 **내가 혼자 처음부터 다시 설계하고 구현할 수 있도록** 만든 공부 로드맵이다.

README.md가 프로젝트의 연구 목표와 설계 배경을 설명한다면, 이 문서는 다음 질문에 답한다.

- 나는 무엇을 공부해야 하는가?
- 어떤 순서로 공부하고 구현해야 하는가?
- 각 단계에서 어떤 개념이 쓰이는가?
- 현재 코드에서는 그 개념이 어디에 구현되어 있는가?
- 다시 처음부터 만든다면 어떤 순서로 시스템을 쌓아야 하는가?

---

## 1. 공부해야 할 것 전체 목록

아래는 FLETCHER를 이해하고 다시 구현하기 위해 공부해야 할 전체 항목이다. 아직 분류하지 않고 먼저 전부 나열한다.

- Python project structure
- Python package / module import
- virtual environment
- `pyproject.toml`
- Git / GitHub workflow
- local development workflow
- smoke test
- CLI script
- GPU가 필요한 이유
- CPU RAM vs. GPU VRAM
- CUDA
- HuggingFace Transformers
- tokenizer
- chat template
- causal language model
- local LLM inference
- open-source LLM
- Qwen
- Llama
- model loading
- model serving
- quantization
- BitsAndBytes
- 4-bit quantization
- context window
- prompt tokens
- completion tokens
- latency
- KV cache
- Flash Attention
- Paged Attention
- vLLM
- concurrent serving
- asyncio
- GIL
- Docker
- Kubernetes
- Colab
- Unsloth
- LoRA
- QLoRA
- fine-tuning
- cold-start SFT
- post-training
- reward design
- RL
- RLHF
- RLFH terminology check
- GRPO
- SAGE
- SkillRL
- sequential rollout
- recursive skill-policy co-evolution
- LLM abstraction
- adapter pattern
- fake backend
- HF backend
- prompt engineering
- prompt contract
- structured output
- JSON output parsing
- Pydantic
- schema validation
- agent communication schema
- critic agent
- content fidelity evaluation
- conceptual correctness
- procedural correctness
- completeness checking
- role specialization
- persona prompting
- strict / merciful / neutral personas
- sycophancy
- self-sycophancy
- hallucination
- parametric knowledge
- self-critique
- debate
- multi-agent system
- Multi-Agent Orchestration system
- LangGraph
- state graph
- shared state
- state isolation
- message history
- race condition
- mutex
- orchestrator
- disagreement detection
- consensus detection
- majority vote
- judge agent
- synthesizer
- early stopping
- debate round depth
- R axis: role diversity
- N axis: persona diversity
- K axis: debate rounds
- S axis: skill usage
- M axis: model diversity
- NM combined sweep
- AI Evaluation pipeline
- hard negative dataset
- normal dataset
- false positive
- recall
- accuracy
- latency measurement
- token measurement
- LLM call count
- benchmark design
- experiment sweep
- reproducibility
- random seed
- master seed
- evaluation log
- JSONL
- audit log
- AI agent with audit logs
- debate trajectory
- experience distillation
- SkillBank
- general skill
- task-specific skill
- skill retrieval
- skill injection
- skill reuse reward
- anomaly penalty
- RAG
- grounding
- embeddings
- SentenceTransformer
- vector similarity
- FAISS
- Vector DB
- top-k retrieval
- chunking
- chunk overlap
- retrieved context
- Secure RAG system with guardrails
- source allowlist
- citation validation
- prompt injection defense
- retrieval confidence threshold
- context filtering
- NoteWriter
- personalization
- style transfer
- persona-preserving prompting
- few-shot tone reproduction
- MCP-Powered AI assistant
- tool-using assistant
- external document connectors
- Knowledge Graph-powered AI
- concept graph
- prerequisite graph
- misconception graph
- LLMOps
- observability
- model registry
- experiment tracking
- deployment
- UI
- Gradio
- Streamlit
- React

---

## 2. 중요도별 공부 우선순위

### ★★★★★ 반드시 먼저 알아야 하는 것

- LLM abstraction
- prompt contract
- structured output
- Pydantic schema
- content critic
- self-critique
- multi-agent debate
- orchestrator
- disagreement / consensus detection
- R / N / K / S experiment variables
- AI Evaluation pipeline
- hard negative dataset
- normal dataset
- latency / token / LLM call metrics
- debate logs
- SkillBank
- RAG / grounding
- embeddings
- Vector DB
- local LLM inference
- quantization

### ★★★★☆ 구현하면서 깊게 이해해야 하는 것

- LangGraph
- role specialization
- persona prompting
- message history
- state isolation
- majority vote
- synthesizer
- context window
- skill retrieval
- experience distillation
- LLMOps
- vLLM
- KV cache
- Flash Attention
- Paged Attention
- LoRA / QLoRA
- cold-start SFT
- GRPO
- RLHF

### ★★★☆☆ 확장 단계에서 공부할 것

- Secure RAG guardrails
- source validation
- prompt injection defense
- MCP-powered assistant
- Knowledge Graph-powered AI
- Docker
- Kubernetes
- experiment tracking
- model registry
- UI
- deployment

### ★★☆☆☆ 지금은 개념만 알고 넘어가도 되는 것

- AutoGen
- ReAct
- advanced judge-agent design
- full production observability
- multi-GPU orchestration
- Kubernetes-based critic isolation

---

## 3. 카테고리별 공부 묶음

### A. 개발 환경과 오픈소스 LLM 실행

중요도: ★★★★★

공부할 것:

- Python project structure
- `pyproject.toml`
- local venv
- Git / GitHub
- GPU, CUDA, VRAM
- HuggingFace Transformers
- tokenizer
- model loading
- quantization
- BitsAndBytes
- local inference
- Colab
- Unsloth

이 프로젝트에서 쓰인 곳:

- `pyproject.toml`
- `fletcher/llm/client.py`
- `fletcher/llm/factory.py`
- `fletcher/llm/hf_client.py`
- `fletcher/llm/fake_client.py`

현재 구현 상태:

- LLM abstraction 구현됨
- fake backend 구현됨
- HuggingFace local backend 구현됨
- 4-bit quantized model name 사용
- 실제 fine-tuning은 아직 future work

### B. LLM Application 기본 구조

중요도: ★★★★★

공부할 것:

- LLMClient abstraction
- adapter pattern
- message format
- prompt contract
- JSON response contract
- token counting
- latency measurement
- fake client testing

이 프로젝트에서 쓰인 곳:

- `fletcher/llm/client.py`
- `fletcher/llm/message.py`
- `fletcher/llm/factory.py`
- `fletcher/llm/fake_client.py`
- `fletcher/llm/hf_client.py`

코드 리뷰 포인트:

- agent들은 HuggingFace를 직접 호출하지 않는다.
- 모든 agent는 `LLMClient.generate()`만 호출한다.
- fake backend는 smoke test를 위해 실제 LLM output contract를 흉내낸다.

### C. Structured Output과 Agent Schema

중요도: ★★★★★

공부할 것:

- Pydantic
- structured output
- JSON parsing
- schema validation
- agent communication protocol
- output failure handling

이 프로젝트에서 쓰인 곳:

- `fletcher/agents/schemas.py`
- `fletcher/agents/content_critic/conceptual.py`
- `fletcher/agents/content_critic/procedural.py`
- `fletcher/agents/content_critic/completeness.py`
- `fletcher/architectures/self_critique.py`

코드 리뷰 포인트:

- 모든 critic은 `CriticVerdict`로 수렴한다.
- debate mode에서는 `message_to_others`가 추가된다.
- 이 schema가 debate log와 future training data의 기반이 된다.

### D. Content Fidelity Critic 설계

중요도: ★★★★★

공부할 것:

- content fidelity evaluation
- hallucination
- parametric knowledge
- sycophancy
- self-sycophancy
- misconception detection
- hard negative design
- role specialization

이 프로젝트에서 쓰인 곳:

- `fletcher/agents/content_critic/conceptual.py`
- `fletcher/agents/content_critic/procedural.py`
- `fletcher/agents/content_critic/completeness.py`
- `eval/datasets/hard_negative/hard_negatives.json`
- `eval/datasets/normal/normal_explanations.json`

코드 리뷰 포인트:

- Conceptual Critic은 정의와 개념을 본다.
- Procedural Critic은 절차와 순서를 본다.
- Completeness Critic은 빠진 핵심 개념을 본다.
- 이 역할 분리가 R-axis다.

### E. Multi-Agent Orchestration System

중요도: ★★★★★

공부할 것:

- multi-agent system
- orchestration pattern
- LangGraph
- state graph
- shared state
- message history
- same-target debate
- role aggregation
- disagreement detection
- consensus detection
- majority vote
- early stopping

이 프로젝트에서 쓰인 곳:

- `fletcher/agents/orchestrator.py`
- `fletcher/architectures/persona_debate.py`
- `fletcher/architectures/debate.py`
- `fletcher/architectures/full_debate.py`
- `fletcher/agents/synthesizer.py`

코드 리뷰 포인트:

- 같은 target을 보는 critic들은 debate할 수 있다.
- 다른 role을 가진 critic들은 서로 debate하는 것이 아니라 synthesize한다.
- `DebateOrchestrator`는 round, disagreement, metrics, debate log를 관리한다.
- `full_debate.py`는 Stage 1 debate와 Stage 2 synthesis를 연결한다.

### F. AI Evaluation Pipeline

중요도: ★★★★★

공부할 것:

- benchmark design
- hard negative dataset
- normal dataset
- accuracy
- recall
- false positive
- latency
- prompt tokens
- completion tokens
- LLM calls
- sweep experiment
- reproducibility
- JSONL logging

이 프로젝트에서 쓰인 곳:

- `eval/run_comparison.py`
- `eval/metrics.py`
- `eval/datasets/`
- `eval/debate_logs/`

코드 리뷰 포인트:

- `critic_fn`이 공통 eval interface다.
- hard negative는 오류 탐지 능력을 본다.
- normal dataset은 과잉 탐지를 본다.
- R/N/K/S/M/NM sweep은 어떤 요소가 성능에 영향을 주는지 분리해서 보기 위한 장치다.

### G. RAG, Embeddings, Vector DB

중요도: ★★★★☆

공부할 것:

- RAG
- grounding
- embeddings
- SentenceTransformer
- chunking
- chunk overlap
- vector similarity
- Vector DB
- FAISS
- top-k retrieval
- context injection
- context window

이 프로젝트에서 쓰인 곳:

- `fletcher/rag/lecture_notes/retriever.py`
- `fletcher/agents/content_critic/conceptual.py`
- `fletcher/agents/content_critic/procedural.py`
- `fletcher/agents/content_critic/completeness.py`
- `eval/run_comparison.py`

코드 리뷰 포인트:

- RAG는 critic의 parametric knowledge만 믿지 않기 위한 장치다.
- retrieved passage는 prompt에 grounding evidence로 들어간다.
- 현재 Vector DB는 FAISS in-memory prototype이다.
- future work는 persistent Vector DB와 Secure RAG guardrails다.

### H. Secure RAG System with Guardrails

중요도: ★★★☆☆

공부할 것:

- source allowlist
- citation validation
- prompt injection defense
- retrieved context filtering
- grounding confidence threshold
- unsafe retrieval handling
- context poisoning

이 프로젝트에서 쓰일 곳:

- `fletcher/rag/lecture_notes/retriever.py`
- `fletcher/agents/content_critic/`
- future guardrail module

현재 구현 상태:

- retrieval prototype은 있음
- 보안 guardrail은 아직 없음

### I. Skill-Based Self-Improvement

중요도: ★★★★★

공부할 것:

- AI agent with audit logs
- debate trajectory
- experience distillation
- SkillBank
- general skill
- task-specific skill
- skill retrieval
- skill injection
- SkillRL
- SAGE
- reward design
- skill reuse reward
- anomaly penalty

이 프로젝트에서 쓰인 곳:

- `fletcher/skills/skill_bank.py`
- `fletcher/skills/retrieval.py`
- `fletcher/skills/distillation.py`
- `fletcher/architectures/skill_augmented.py`
- `scripts/distill_debate_logs.py`
- `fletcher/skills/skill_bank.jsonl`

코드 리뷰 포인트:

- debate log가 experience source다.
- distillation이 log를 skill로 바꾼다.
- SkillBank는 JSONL로 저장된다.
- 현재 retrieval은 token overlap 기반이다.
- future work는 embedding retrieval이다.

### J. Fine-Tuned Open-Source LLM

중요도: ★★★★☆

공부할 것:

- open-source LLM fine-tuning
- Unsloth
- LoRA
- QLoRA
- cold-start SFT
- supervised fine-tuning dataset
- reward model 없이 하는 policy optimization
- GRPO
- RLHF
- RLFH terminology check

이 프로젝트에서 쓰인 곳:

- `fletcher/finetuning/policy_update.py`
- `fletcher/llm/hf_client.py`
- future Unsloth training script

코드 리뷰 포인트:

- 현재는 실제 학습은 없다.
- `policy_update.py`는 debate log를 reward/training record로 바꾸는 준비 단계다.
- SkillRL 관점에서는 cold-start SFT 후 GRPO로 가는 흐름이다.

### K. LLMOps와 Serving

중요도: ★★★★☆

공부할 것:

- model serving
- latency
- throughput
- concurrency
- asyncio
- vLLM
- KV cache
- Flash Attention
- Paged Attention
- quantization
- monitoring
- experiment tracking
- model registry
- deployment

이 프로젝트에서 쓰인 곳:

- `fletcher/serving/config.py`
- `fletcher/llm/hf_client.py`
- `eval/metrics.py`
- `eval/run_comparison.py`

코드 리뷰 포인트:

- 현재 debate execution은 sequential이다.
- `serving/config.py`는 future serving mode를 설계하기 위한 자리다.
- 실제 vLLM runtime은 아직 구현되지 않았다.

### L. MCP-Powered AI Assistant

중요도: ★★★☆☆

공부할 것:

- MCP
- tool-using assistant
- external connectors
- document retrieval
- calendar / email / LMS / note app integration
- tool schema
- permission boundary

이 프로젝트와의 관계:

- FLETCHER critic pipeline을 MCP tool로 노출할 수 있다.
- lecture notes, student notes, LMS 자료를 MCP connector로 가져올 수 있다.

현재 구현 상태:

- 아직 구현되지 않음
- future product integration 주제

### M. Knowledge Graph-Powered AI

중요도: ★★★☆☆

공부할 것:

- knowledge graph
- concept graph
- prerequisite graph
- misconception graph
- graph retrieval
- graph RAG

이 프로젝트와의 관계:

- 강의 개념을 node로 만들 수 있다.
- prerequisite 관계를 edge로 만들 수 있다.
- misconception과 correction skill을 graph로 연결할 수 있다.

현재 구현 상태:

- 아직 구현되지 않음
- future knowledge layer 주제

---

## 4. 혼자 처음부터 다시 만드는 시스템 설계 순서

이 섹션이 가장 중요하다. FLETCHER를 처음부터 다시 만든다면 아래 순서로 진행한다.

---

### Step 0. 문제 정의와 실험 질문 확정

목표:

- 이 프로젝트가 무엇을 검증하려는지 정한다.
- 단순 챗봇이 아니라 content fidelity critic임을 명확히 한다.

공부할 것:

- content fidelity evaluation
- hallucination
- sycophancy
- hard negative evaluation
- self-critique vs. debate

프로젝트에서 어디에 해당하는가:

- `README.md`
- `eval/datasets/hard_negative/hard_negatives.json`
- `eval/datasets/normal/normal_explanations.json`

구현할 것:

- hard negative dataset 초안
- normal dataset 초안
- Architecture 1, 2, 3 비교 목표

완료 기준:

- 내가 무엇을 측정할지 말할 수 있어야 한다.
- `flagged=True/False`가 어떤 의미인지 정의되어 있어야 한다.

---

### Step 1. Python 프로젝트 뼈대 만들기

목표:

- import 가능한 Python package를 만든다.
- 나중에 agent, architecture, eval이 섞이지 않도록 디렉터리를 나눈다.

공부할 것:

- Python package
- `__init__.py`
- `pyproject.toml`
- local venv
- Git
- smoke script

현재 코드:

- `pyproject.toml`
- `fletcher/__init__.py`
- `fletcher/`
- `eval/`
- `scripts/`

구현할 것:

```text
fletcher/
├── llm/
├── agents/
├── architectures/
├── rag/
├── skills/
├── finetuning/
└── serving/

eval/
scripts/
```

완료 기준:

- `PYTHONPATH=. python3 ...`로 로컬 패키지를 import할 수 있다.

---

### Step 2. LLM abstraction 만들기

목표:

- agent가 특정 LLM backend에 종속되지 않게 만든다.

공부할 것:

- interface
- adapter pattern
- HuggingFace Transformers
- fake backend
- prompt tokens
- completion tokens
- latency

현재 코드:

- `fletcher/llm/client.py`
- `fletcher/llm/message.py`
- `fletcher/llm/factory.py`
- `fletcher/llm/fake_client.py`
- `fletcher/llm/hf_client.py`

구현할 것:

- `Message`
- `GenerationConfig`
- `LLMResponse`
- `LLMClient`
- `FakeLLMClient`
- `HFLocalClient`
- `create_llm_client`

완료 기준:

- fake backend와 HF backend를 같은 interface로 호출할 수 있다.
- agent 코드는 HuggingFace import를 몰라도 된다.

---

### Step 3. Structured output schema 만들기

목표:

- critic이 항상 같은 형태로 답하게 한다.

공부할 것:

- Pydantic
- JSON parsing
- structured output
- schema validation
- prompt contract

현재 코드:

- `fletcher/agents/schemas.py`

구현할 것:

```text
CriticVerdict
├── role
├── flagged
├── reasoning
├── confidence
└── message_to_others
```

완료 기준:

- critic output이 dict가 아니라 `CriticVerdict`로 검증된다.
- debate log에 바로 저장할 수 있다.

---

### Step 4. Architecture 1: Self-Critique baseline 만들기

목표:

- multi-agent debate와 비교할 single-agent baseline을 만든다.

공부할 것:

- self-critique
- sycophancy
- verdict parsing
- JSON self-review
- stopping threshold

현재 코드:

- `fletcher/architectures/self_critique.py`
- `fletcher/pipeline.py`

구현할 것:

- `SelfCritique1Pass`
- `SelfCritique`
- `VERDICT: FLAGGED / OK`
- review confidence
- revise loop

완료 기준:

- hard negative와 normal sample을 넣고 `last_flagged`가 나온다.
- adaptive version이 threshold로 early stop할 수 있다.

---

### Step 5. Critic role 세분화하기

목표:

- 하나의 critic이 모든 것을 보는 대신 역할을 나눈다.

공부할 것:

- role specialization
- conceptual correctness
- procedural correctness
- completeness checking
- hallucination
- grounding

현재 코드:

- `fletcher/agents/content_critic/conceptual.py`
- `fletcher/agents/content_critic/procedural.py`
- `fletcher/agents/content_critic/completeness.py`

구현할 것:

- `ConceptualCritic`
- `ProceduralCritic`
- `CompletenessCritic`
- persona prompt
- optional retriever
- `CriticVerdict` parsing

완료 기준:

- 각 critic이 같은 explanation을 다른 기준으로 평가한다.
- role이 `conceptual`, `procedural`, `completeness`로 분리된다.

---

### Step 6. Persona axis 만들기

목표:

- 같은 role 안에서도 판단 성향이 다른 critic을 만든다.

공부할 것:

- persona prompting
- strict vs. merciful
- N-axis
- self-sycophancy
- disagreement induction

현재 코드:

- `PERSONA_PROMPTS` in `conceptual.py`
- `PERSONA_PROMPTS` in `procedural.py`
- `fletcher/architectures/persona_debate.py`

구현할 것:

- `strict`
- `merciful`
- `neutral`
- role + persona critic key

완료 기준:

- `conceptual_strict`, `conceptual_merciful` 같은 critic key가 생긴다.
- N=1, N=2, N=3 비교가 가능해진다.

---

### Step 7. DebateOrchestrator 만들기

목표:

- 같은 target을 보는 critic들이 서로 판단을 검토하게 만든다.

공부할 것:

- Multi-Agent Orchestration system
- debate round
- message history
- disagreement detection
- consensus detection
- majority vote
- early stopping
- audit logs

현재 코드:

- `fletcher/agents/orchestrator.py`

구현할 것:

- `DebateCriticSpec`
- `DebateOrchestrator`
- round loop
- `debate_history`
- `_has_disagreement`
- final majority vote
- metrics accumulation
- debate log output

완료 기준:

- N명의 critic이 round별로 평가한다.
- disagreement가 없으면 early stop한다.
- final result와 debate log가 나온다.

---

### Step 8. Role aggregation과 Synthesizer 만들기

목표:

- 서로 다른 role 결과를 하나의 final feedback으로 합친다.

공부할 것:

- role aggregation
- synthesizer
- R-axis
- why different-role critics do not debate

현재 코드:

- `fletcher/agents/synthesizer.py`
- `fletcher/architectures/debate.py`
- `fletcher/architectures/full_debate.py`

구현할 것:

- `Synthesizer.synthesize`
- role verdict aggregation
- final result text
- `issue_found`
- `flagged_roles`

완료 기준:

- conceptual issue와 procedural issue를 동시에 담을 수 있다.
- role 간 결과를 majority vote로 뭉개지 않는다.

---

### Step 9. Architecture 2 전체 파이프라인 만들기

목표:

- Stage 1 debate와 Stage 2 synthesis를 연결한다.

공부할 것:

- nested orchestration
- R-axis
- N-axis
- K-axis
- full architecture composition

현재 코드:

- `fletcher/architectures/persona_debate.py`
- `fletcher/architectures/debate.py`
- `fletcher/architectures/full_debate.py`

구현할 것:

```text
for each role:
    run same-role persona debate
    choose per-role majority verdict

synthesize role verdicts
return final result + log
```

완료 기준:

- `run_full_debate()`가 role list를 받아 전체 Architecture 2를 실행한다.
- debate log 안에 role별 log가 들어간다.

---

### Step 10. RAG retriever 만들기

목표:

- critic이 model memory만 믿지 않고 source material을 참조하게 만든다.

공부할 것:

- RAG
- grounding
- embeddings
- SentenceTransformer
- FAISS
- Vector DB
- chunking
- top-k retrieval
- context window

현재 코드:

- `fletcher/rag/lecture_notes/retriever.py`

구현할 것:

- source text fetch
- text cleaning
- chunking
- embedding
- FAISS index
- retrieve(query)

완료 기준:

- explanation을 query로 넣으면 reference passages가 반환된다.
- critic prompt에 retrieved passages가 들어간다.

---

### Step 11. Secure RAG guardrails 설계하기

목표:

- retrieved context가 오염되거나 공격적인 prompt가 되는 것을 막는다.

공부할 것:

- prompt injection
- source allowlist
- citation validation
- context filtering
- retrieval confidence threshold
- grounding verification

현재 코드:

- 아직 전용 guardrail module 없음
- RAG prototype은 `fletcher/rag/lecture_notes/retriever.py`

구현할 것:

- allowed source registry
- retrieved text sanitizer
- citation metadata
- low-confidence retrieval fallback
- prompt injection detector

완료 기준:

- critic prompt에 들어가기 전 retrieved context를 검증한다.

---

### Step 12. Evaluation pipeline 만들기

목표:

- architecture별 성능을 같은 기준으로 비교한다.

공부할 것:

- AI Evaluation pipeline
- benchmark dataset
- hard negative
- normal sample
- accuracy
- false positive
- latency
- token usage
- LLM call count
- JSONL logs

현재 코드:

- `eval/metrics.py`
- `eval/run_comparison.py`
- `eval/datasets/`

구현할 것:

- `EvalResult`
- `EvalSummary`
- `evaluate_dataset`
- `print_summary`
- `run_comparison.py`
- `--smoke`
- `--log-debates`

완료 기준:

- fake backend로 전체 smoke가 돈다.
- architecture별 summary가 출력된다.
- debate-capable run은 JSONL log를 남긴다.

---

### Step 13. Experiment sweep 설계하기

목표:

- 어떤 요소가 성능에 영향을 주는지 분리해서 측정한다.

공부할 것:

- controlled experiment
- independent variable
- dependent variable
- ablation
- reproducibility
- seed

현재 코드:

- `eval/run_comparison.py`

구현할 축:

- Architecture: 1, 2, 3
- R: role diversity
- N: persona diversity
- K: debate rounds
- S: skill usage
- M: model diversity
- NM: persona + model combined

완료 기준:

- R/N/K/S/M/NM sweep을 따로 실행할 수 있다.
- S_on과 S_off의 token overhead를 비교할 수 있다.

---

### Step 14. Debate logs와 audit trail 만들기

목표:

- agent가 왜 그런 결정을 했는지 나중에 추적 가능하게 만든다.

공부할 것:

- audit log
- JSONL
- trace
- debate trajectory
- reproducibility
- evaluation provenance

현재 코드:

- `eval/metrics.py`
- `eval/debate_logs/`
- `fletcher/agents/orchestrator.py`

구현할 것:

- per-round verdict
- final decision
- expected label
- predicted label
- correctness
- metrics
- skill metrics

완료 기준:

- 하나의 sample에 대해 critic별 판단 과정을 log로 볼 수 있다.

---

### Step 15. SkillBank와 experience distillation 만들기

목표:

- debate 경험을 버리지 않고 다음 평가에 재사용한다.

공부할 것:

- SkillRL
- SAGE
- experience distillation
- SkillBank
- general skill
- task-specific skill
- success / failure trajectory

현재 코드:

- `fletcher/skills/skill_bank.py`
- `fletcher/skills/distillation.py`
- `scripts/distill_debate_logs.py`

구현할 것:

- `Skill`
- `SkillBank`
- `distill_skills_from_debate_log`
- `--reset`
- stable skill id

완료 기준:

- debate log를 SkillBank JSONL로 바꿀 수 있다.

---

### Step 16. Skill retrieval과 Architecture 3 만들기

목표:

- 과거 skill을 현재 critic prompt에 넣어 성능 변화를 측정한다.

공부할 것:

- retrieval
- semantic similarity
- token overlap baseline
- skill injection
- context window
- S-axis
- token overhead

현재 코드:

- `fletcher/skills/retrieval.py`
- `fletcher/architectures/skill_augmented.py`
- `eval/run_comparison.py`

구현할 것:

- `SkillRetriever`
- `format_skills_for_prompt`
- `SkillAugmentedCritic`
- `run_skill_persona_debate`
- S sweep

완료 기준:

- S_off와 S_on을 비교할 수 있다.
- retrieved skill count와 retrieved skill tokens가 log에 남는다.

---

### Step 17. Reward preparation 만들기

목표:

- 나중에 fine-tuning / GRPO에 쓸 reward record를 만든다.

공부할 것:

- reward design
- outcome reward
- skill reuse reward
- schema validity reward
- anomaly penalty
- GRPO
- RLHF
- cold-start SFT

현재 코드:

- `fletcher/finetuning/policy_update.py`

구현할 것:

- `RewardWeights`
- `compute_skill_reward`
- `build_training_records`

완료 기준:

- debate log를 reward가 포함된 training record로 바꿀 수 있다.

---

### Step 18. Open-source LLM fine-tuning 준비하기

목표:

- critic이 skill을 더 잘 쓰도록 학습시킬 준비를 한다.

공부할 것:

- Unsloth
- LoRA
- QLoRA
- SFT
- dataset formatting
- instruction tuning
- GRPO
- RLHF
- quantization-aware fine-tuning

현재 코드:

- `fletcher/finetuning/`
- `fletcher/llm/hf_client.py`

구현할 것:

- SFT dataset exporter
- LoRA training script
- validation script
- checkpoint swap logic

완료 기준:

- SkillBank 사용 demonstration을 SFT dataset으로 만들 수 있다.

---

### Step 19. LLMOps와 serving 설계하기

목표:

- multi-agent critic을 실제로 빠르게 serving할 방법을 설계한다.

공부할 것:

- vLLM
- KV cache
- Paged Attention
- Flash Attention
- batching
- concurrency
- GPU parallelism
- latency / throughput tradeoff
- quantization

현재 코드:

- `fletcher/serving/config.py`
- `fletcher/llm/hf_client.py`
- `eval/metrics.py`

구현할 것:

- serving mode config
- latency estimator
- async critic execution
- vLLM backend

완료 기준:

- sequential vs. parallel vs. vLLM serving을 비교할 수 있다.

---

### Step 20. MCP-powered assistant로 확장하기

목표:

- FLETCHER를 독립 script가 아니라 tool-using assistant로 확장한다.

공부할 것:

- MCP
- tool schema
- connector
- permissions
- external documents
- Google Drive / Notion / LMS / local notes integration

현재 코드:

- 아직 없음
- 연결 후보는 `fletcher/pipeline.py`, `fletcher/architectures/full_debate.py`

구현할 것:

- FLETCHER critique tool
- document retrieval tool
- note-writing tool
- user profile / tone sample tool

완료 기준:

- 외부 문서나 노트를 가져와 FLETCHER critic pipeline에 넣을 수 있다.

---

### Step 21. Knowledge Graph-powered AI로 확장하기

목표:

- lecture concepts, prerequisite, misconception, skill을 graph로 연결한다.

공부할 것:

- Knowledge Graph
- graph schema
- entity extraction
- relation extraction
- graph RAG
- prerequisite graph

현재 코드:

- 아직 없음

구현할 graph:

- concept node
- lecture section node
- misconception node
- skill node
- prerequisite edge
- contradicts edge
- corrected-by edge
- appears-in edge

완료 기준:

- critic이 retrieved passages뿐 아니라 concept relationship도 참조할 수 있다.

---

### Step 22. UI와 product layer 만들기

목표:

- 사용자가 설명을 넣고 personalized feedback을 받을 수 있게 한다.

공부할 것:

- Gradio
- Streamlit
- React
- API server
- session state
- user profile
- tone sample storage

현재 코드:

- 아직 없음
- 현재 user-facing entry는 `fletcher/pipeline.py`

구현할 것:

- input form
- tone sample input
- critic result view
- note output view
- debate log viewer

완료 기준:

- CLI가 아니라 UI로 FLETCHER를 사용할 수 있다.

---

## 5. 현재 코드 리뷰 순서

처음부터 읽는다면 이 순서로 읽는다.

### 1순위: 시스템의 입구

1. `fletcher/pipeline.py`
2. `fletcher/llm/client.py`
3. `fletcher/llm/factory.py`
4. `fletcher/llm/fake_client.py`
5. `fletcher/llm/hf_client.py`

리뷰 질문:

- LLM backend는 어떻게 교체되는가?
- fake와 HF backend는 같은 interface를 만족하는가?
- token/latency는 어디서 기록되는가?

### 2순위: agent schema와 critic

1. `fletcher/agents/schemas.py`
2. `fletcher/agents/content_critic/conceptual.py`
3. `fletcher/agents/content_critic/procedural.py`
4. `fletcher/agents/content_critic/completeness.py`

리뷰 질문:

- 모든 critic output은 같은 schema인가?
- role별 책임이 섞이지 않는가?
- RAG context는 어디에 들어가는가?
- debate history는 어디에 들어가는가?

### 3순위: debate system

1. `fletcher/agents/orchestrator.py`
2. `fletcher/architectures/persona_debate.py`
3. `fletcher/agents/synthesizer.py`
4. `fletcher/architectures/debate.py`
5. `fletcher/architectures/full_debate.py`

리뷰 질문:

- disagreement는 어떻게 정의되는가?
- consensus가 생기면 어떻게 멈추는가?
- final decision rule은 무엇인가?
- Stage 1 debate와 Stage 2 synthesis는 어떻게 다르게 동작하는가?

### 4순위: evaluation

1. `eval/metrics.py`
2. `eval/run_comparison.py`
3. `eval/datasets/hard_negative/hard_negatives.json`
4. `eval/datasets/normal/normal_explanations.json`

리뷰 질문:

- `critic_fn` interface는 무엇인가?
- hard negative와 normal dataset의 label 의미는 무엇인가?
- R/N/K/S/M/NM sweep은 어디서 정의되는가?
- debate log는 어떤 shape으로 저장되는가?

### 5순위: RAG와 SkillBank

1. `fletcher/rag/lecture_notes/retriever.py`
2. `fletcher/skills/skill_bank.py`
3. `fletcher/skills/retrieval.py`
4. `fletcher/skills/distillation.py`
5. `fletcher/architectures/skill_augmented.py`
6. `scripts/distill_debate_logs.py`

리뷰 질문:

- retrieved context는 prompt에 어떻게 들어가는가?
- Skill은 어떤 fields를 가지는가?
- debate log는 어떻게 skill로 바뀌는가?
- S_on은 S_off보다 어떤 token overhead를 만드는가?

### 6순위: future training / serving

1. `fletcher/finetuning/policy_update.py`
2. `fletcher/serving/config.py`

리뷰 질문:

- reward는 어떻게 계산되는가?
- skill reuse reward는 무엇인가?
- sequential serving과 parallel serving latency estimate는 어떻게 다른가?

---

## 6. 프로젝트 진행 순서 요약

혼자 다시 만들 때는 아래 순서로 진행한다.

1. Problem framing
2. Dataset skeleton
3. Python package skeleton
4. LLM abstraction
5. Fake backend
6. HF backend
7. Critic schema
8. SelfCritique baseline
9. Conceptual / Procedural / Completeness critics
10. Persona prompts
11. DebateOrchestrator
12. Synthesizer
13. Architecture 2 full debate
14. RAG retriever
15. Eval metrics
16. Eval sweep runner
17. Debate logs
18. SkillBank
19. Skill distillation
20. Skill retrieval
21. Skill-augmented debate
22. S sweep
23. Reward preparation
24. Fine-tuning preparation
25. Serving optimization
26. MCP extension
27. Knowledge Graph extension
28. UI / product layer

---

## 7. 각 단계에서 실행해볼 명령어

Fake backend 전체 smoke:

```bash
PYTHONPATH=. python3 eval/run_comparison.py --backend fake --smoke --log-debates
```

최신 debate log를 SkillBank로 재생성:

```bash
PYTHONPATH=. python3 scripts/distill_debate_logs.py --input eval/debate_logs/<log_file>.jsonl --reset
```

Skill pipeline 확인:

```bash
PYTHONPATH=. python3 scripts/test_skill_pipeline.py
```

Full debate 확인:

```bash
PYTHONPATH=. python3 scripts/test_full_debate.py
```

Reward / serving helper 확인:

```bash
PYTHONPATH=. python3 scripts/test_policy_and_serving.py
```

---

## 8. 내가 직접 다시 구현할 때 체크리스트

- [ ] 문제 정의를 한 문장으로 설명할 수 있다.
- [ ] Architecture 1, 2, 3의 차이를 설명할 수 있다.
- [ ] R, N, K, S, M, NM이 무엇인지 설명할 수 있다.
- [ ] LLM abstraction을 직접 만들 수 있다.
- [ ] fake backend를 만들 수 있다.
- [ ] HuggingFace backend를 만들 수 있다.
- [ ] Pydantic schema를 만들 수 있다.
- [ ] ConceptualCritic을 만들 수 있다.
- [ ] ProceduralCritic을 만들 수 있다.
- [ ] CompletenessCritic을 만들 수 있다.
- [ ] SelfCritique baseline을 만들 수 있다.
- [ ] persona prompt를 만들 수 있다.
- [ ] DebateOrchestrator를 만들 수 있다.
- [ ] disagreement detection을 구현할 수 있다.
- [ ] early stopping을 구현할 수 있다.
- [ ] majority vote를 구현할 수 있다.
- [ ] Synthesizer를 만들 수 있다.
- [ ] full debate pipeline을 연결할 수 있다.
- [ ] RAG retriever를 만들 수 있다.
- [ ] eval dataset을 만들 수 있다.
- [ ] metrics를 계산할 수 있다.
- [ ] JSONL debate log를 저장할 수 있다.
- [ ] debate log를 SkillBank로 distill할 수 있다.
- [ ] SkillRetriever를 만들 수 있다.
- [ ] S_on / S_off sweep을 비교할 수 있다.
- [ ] reward record를 만들 수 있다.
- [ ] LoRA / QLoRA fine-tuning 계획을 설명할 수 있다.
- [ ] vLLM serving이 왜 필요한지 설명할 수 있다.
- [ ] MCP extension이 어디에 붙는지 설명할 수 있다.
- [ ] Knowledge Graph가 어떤 정보를 표현할지 설명할 수 있다.

---

## 9. 이 프로젝트를 한 문장으로 설명하기

FLETCHER는 학생의 설명을 여러 전문 critic agent가 검토하고, 필요하면 debate를 통해 판단을 보정하며, RAG로 근거를 보강하고, debate log를 SkillBank로 distill해서 다음 평가에 재사용하는 **multi-agent, RAG-grounded, evaluation-driven, self-improving AI learning assistant prototype**이다.
