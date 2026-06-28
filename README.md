# FLETCHER

FLETCHER는 텍스트 기반의 학습 에이전트이다. 사용자는 전공 수업 내용을 영어로 설명하면, debate 기반의 멀티 critic 시스템이 그 설명을 grounding된 근거로 검증하고 반박하여 내용의 충실성을 평가한 뒤 사용자의 말투로 복습 노트를 작성한다.

---

## 1. Project Goal

목표는 다음과 같다.

1. 연구 질문 — debate 기반 멀티에이전트 critique가 single-prompt critique보다 내용 충실성 평가에서 실제로 더 정확한가, 그 대가로 응답 시간(latency)과 토큰 사용량을 얼마나 지불하는가 — 를 정량 비교하는 eval harness를 프로젝트의 1급 산출물로 삼는다.
2. critic의 수와 다양성을 늘리는 것이 항상 더 좋은가, 아니면 어느 지점부터 오히려 성능이 떨어지는가에 대한 정량적인 답을 점진적으로 찾아간다.
3. 직접 구현으로 얻는 스킬 습득 — LLM이 어떻게 서빙되고, 가속되고, 파인튜닝 되고, 정렬되는지, 그리고 여러 LLM 인스턴스가 어떻게 서로 토론하는지를 black-box API 호출 없이 처음부터 직접 구현하여 익힌다.

### 1.1 Research Goal

debate 기반 멀티에이전트 critique가 single-prompt 베이스라인보다 내용 충실성 평가 정확도를 얼마나 높이는가, 그 대가로 응답 시간과 토큰 사용량을 얼마나 지불하는가?

**독립변수**:

- **debate round 수** (0 = single-prompt 베이스라인, 1, 2, 3...)
- **critic ensemble 크기 (N)** — 같은 역할의 Content Critic을 몇 명 병렬로 둘 것인가 (다수결/합의 효과)
- **critic 역할 다양성 (R)** — 서로 다른 관점의 critic 역할을 몇 종류 추가할 것인가 (역할 다양성 효과)

**종속변수**:

- 평가 정확도
- 응답 시간
- 토큰 사용량

**연구 진행 순서 (점진적)**:

1. round 수만 고정하고 ensemble 크기(N) 단독으로 스윕 — "같은 역할 여러 명이면 더 정확해지는가"
2. ensemble 크기 고정하고 역할 다양성(R) 단독으로 스윕 — "다른 관점을 추가하면 더 정확해지는가"
3. 1, 2에서 얻은 각각의 최적 지점을 조합해 N과 R을 함께 변화시키며 debate round와의 상호작용까지 — 전체 설계공간(design space)에서 정확도/latency/cost 균형이 가장 좋은 조합을 탐색

### 1.2 Learning Goal

**(0) 환경 구축**

- GPU가 왜, 얼마나 필요한가 — vRAM vs CPU RAM
- CUDA
- HuggingFace (모델 로딩, 파이프라인 구조)
- Unsloth (경량 파인튜닝 프레임워크)
- Colab (무료 T4 → 필요 시 Pro), Docker, 로컬 `venv` + Github

**(3) Multi-agent 구축, 역할 세분화, shared variable**

- LangGraph
- LLM 호출 추상화, prompt 설계
- orchestration pattern
- shared variable / state 격리 — race condition, message history, mutex
- separation of concerns (context, language critic 분리)
- structured output (Pydantic) — critic 간 통신 스키마 정의
- ReAct, AutoGen

**(4) 아첨, 환각 억제**

- critic이 틀린 곳을 탐지하도록 설계 (의도적으로 틀린 설명 주입해서 탐지 여부 확인)
- 소형 모델의 아첨 경향
- prompt 기반 억제 vs fine-tuning 기반 억제
- 아첨의 발생 원인 및 해결책 — self-critique, debate
- 환각의 개념 및 원인 — parametric knowledge
- Unsloth 파인튜닝
- 사후학습(post-training), 강화학습(RL)
- 학습·평가용 데이터 및 데이터 합성

**(5) Debate 및 대기시간 최소화**

- local model limitation
- multi-agent debate, debate structure (critic / judge)
- 추론 지연의 원인 찾기
- 대기 시간 최소화 — asyncio, concurrency, GIL, early stopping
- KV Cache, Flash Attention, Paged Attention
- vLLM, GPU parallelism

**(6) Grounding / RAG**

- RAG, embedding, grounding
- Docker, 쿠버네티스 (멀티 컨테이너로 critic 격리/서빙할 경우)

---

## 2. 핵심 차별점

### 2.1 FLETCHER — Self-Sycophancy를 견제하는 critic

영화 *위플래쉬*의 Fletcher는 "잘했어"를 남발하지 않는다. 다만 FLETCHER에서 이 테마가 막아야 하는 대상은 "사용자에게 아첨"이 아니라 **"critic 자신이 사용자 설명을 대충 보고 안일하게 통과시키는 것(self-sycophancy)"** 이다. critic 한 명만 두면 이걸 잡을 방법이 없다 — critic 여러 명이 서로의 평가를 의심하고 반박하는 구조(debate)가, 한 critic이 안일하게 넘긴 오류를 다른 critic이 잡아내는 메커니즘이 된다.

→ Learning Goal **(4) 아첨, 환각 억제** 중 "아첨의 발생 원인 및 해결책 — self-critique, debate" / **(5) Debate 및 대기시간 최소화** 중 "multi-agent debate, debate structure (critic / judge)"

### 2.2 Grounded Content Critic

전공 내용 정확성을 critic의 parametric knowledge(모수 지식)에만 의존해 판단하면, 사용자의 오개념을 critic이 함께 틀리며 승인하는 환각 채점기가 된다. Content Critic은 강의자료·논문을 RAG로 검색해 ground truth에 비춰 채점한다.

→ Learning Goal **(4) 아첨, 환각 억제** 중 "환각의 개념 및 원인 — parametric knowledge" / **(6) Grounding / RAG** 중 "RAG, embedding, grounding"

### 2.3 토론 — 왜 멀티에이전트인가

"Content Critic이 내용을 보고, Language Coach가 영어를 본 뒤 결과를 그냥 합친다"는 분업(division of labor) 구조는, 결과적으로 LLM을 순차적으로 두 번 부르는 것과 다르지 않다 — 멀티에이전트라서 더 정확해진다는 근거가 되지 못한다. 진짜 멀티에이전트의 가치는 **같은 평가 대상에 대해 의견이 갈렸을 때 서로 반박하며 합의(consensus)에 도달하는 과정** 자체에 있다. 이 차이를 증명하는 것이 연구 목표의 핵심이다.

→ Learning Goal **(3) Multi-agent 구축** 중 "orchestration pattern", "separation of concerns (context, language critic 분리)" / **(5) Debate 및 대기시간 최소화** 중 "multi-agent debate, debate structure (critic / judge)"

---

## 3. System Architecture

### 3.1 Architecture 1 — Self-Critique (single-agent baseline)

```
텍스트 입력 (전공 내용을 영어로 설명)
        │
        ▼
   단일 LLM — 1차 평가 생성
        │
        ▼
   동일 LLM — 자기 출력 재검토 (self-critique, 1 pass)
        │
        ▼
   NoteWriter   (사용자 말투로 복습 노트 작성)
```

1.1에서 정의한 **debate round = 0 베이스라인**. 에이전트가 여러 개가 아니라, 같은 LLM이 자기 출력을 한 번 더 검토하는 구조 — multi-agent가 아니라 single-prompt의 확장판. "critic 여러 명이 필요 없다면 이걸로 충분해야 한다"는 게 비교 기준점.

→ Learning Goal **(4) 아첨, 환각 억제** 중 "아첨의 발생 원인 및 해결책 — self-critique"

### 3.2 Architecture 2 — Multi-agent Debate

```
텍스트 입력
        │
        ▼
   Content Critic × N, 역할 R종   (병렬, 독립적으로 1차 평가)
        │
        ▼
   Orchestrator   (의견 충돌 감지)
        │
   ┌────┴────────────────┐
 합의 도출됨           의견 충돌 있음
   │                     ▼
   │              Debate Round × K (최대 라운드 제한)
   └─────────┬───────────┘
             ▼
       Synthesizer → NoteWriter
```

#### 3.2.1 Role specialization

"내용 정확성"이라는 같은 목표를 다른 관점에서 보는 critic을 몇 종류 둘 것인가. 후보 세 가지를 검토했고, 정당화 강도와 인프라 준비 상태에 따라 단계적으로 도입한다.

**채택 (R=2, MVP부터 적용)**

- **Conceptual Critic** — 개념·정의 자체의 정확성을 본다. 강의자료의 개념 설명 부분을 RAG로 검색해 대조.
- **Procedural Critic** — 개념은 맞아도 적용 절차·단계 순서가 논리적으로 맞는지를 본다. 강의자료의 예제 풀이·증명 과정을 RAG로 검색해 대조.
- 채택 이유: 두 critic이 요구하는 RAG 검색 대상과 평가 방식이 서로 다름 — 역할 분리가 "각자 다른 context가 필요하다"는 기준을 만족시키는 가장 깔끔한 사례.

**보류 (Phase 2 이후, 6번 로드맵에 별도 기재)**

- **Completeness Critic** — 설명에 빠진 내용이 있는지를 전담. 보류 이유: 판단 기준이 단순 문서 검색이 아니라 커리큘럼 수준의 ground truth(목차·개념 그래프)를 필요로 함 — 현재 RAG 인프라(강의자료 2-3개) 수준으로는 부족.

**제외**

- **Clarity Checker** — 설명이 모호해서 채점 자체가 불가능한 수준인지를 체크. 제외 이유: 사실상 기존 LanguageCoach의 부활이며, "영어는 평가 대상이 아니다"(한 줄 정의)와 충돌해 혼란을 야기.

→ Learning Goal **(3) Multi-agent 구축** 중 "separation of concerns (context, language critic 분리)"

#### 3.2.2 Ensemble 크기

같은 역할의 critic을 몇 명 병렬로 둘 것인가 (다수결/합의 효과).

→ Learning Goal **(3)** "orchestration pattern" / **(5)** "GPU parallelism"

#### 3.2.3 Debate depth

의견 충돌 시 몇 라운드까지 반박을 허용할 것인가, 종료 조건은 무엇인가 (합의 도달 시 조기 종료 vs 고정 라운드).

→ Learning Goal **(5) Debate 및 대기시간 최소화** 중 "multi-agent debate, debate structure (critic / judge)", "early stopping"

### 3.3 최종 목표

Architecture 1(self-critique)과 Architecture 2(multi-agent debate, R/N/K 세 축)를 같은 입력으로 돌려 비교하고, **정확도 대비 속도(latency)·비용(token)이 가장 효율적인 지점**을 찾는다. "에이전트가 많을수록 무조건 좋은가"에 대한 답은 이 탐색 과정에서 나온다 — 어느 지점부터 R·N·K를 늘려도 정확도 향상이 없거나 오히려 떨어지는지(사공이 많으면 배가 산으로 가는 지점)를 데이터로 보이는 게 핵심 결과물.

---

## 4. 알아야 할 어려운 문제들

### 4.1 Content Critic grounding 없으면 환각 채점기가 됨

RAG로 강의 PDF·논문을 vector DB에 넣고 ground truth 대조. MVP는 강의자료 몇 개만으로 시작. Conceptual Critic과 Procedural Critic이 검색할 대상이 다르므로(개념 설명 vs 예제 풀이), 같은 vector DB라도 검색 쿼리·청크 단위를 critic별로 다르게 설계해야 함.

→ Learning Goal **(6) Grounding / RAG** 중 "RAG, embedding, grounding"

### 4.2 의견 충돌(disagreement)을 어떻게 정의하고 감지할 것인가

Orchestrator가 "합의됨"과 "충돌 있음"을 가르는 기준이 모호하면 안 됨 — critic들의 평가가 점수 형태라면 임계값(threshold) 차이로 판단할지, 텍스트 평가라면 별도 판정 모델이 필요한지부터 정의해야 함. 이 기준이 느슨하면 항상 debate가 트리거되어 3.3에서 측정하려는 "round=0 vs round>0" 비교 자체가 무의미해지고, 너무 빡빡하면 거의 합의로 처리돼서 멀티에이전트 효과를 측정할 기회가 사라짐.

→ Learning Goal **(3) Multi-agent 구축** 중 "orchestration pattern"

### 4.3 Debate 종료 조건 — 무한 토론 방지

critic들이 토론을 진행하면 할수록 성능이 어떻게 변하는가, 몇 라운드까지가 적당한가(3.2.3 debate depth, K). 종료 조건 후보:

- **고정 라운드**: K를 미리 정해두고 무조건 K번 토론 후 종료 (가장 단순, 비교 실험엔 유리)
- **합의 도달 시 조기 종료**: critic들의 평가가 수렴하면 K 전이라도 멈춤 (실서비스엔 유리하지만, round 수 자체가 변수가 되어버려 실험 통제가 어려워짐)
- **최종 판단자(judge) 도입**: critic들이 끝까지 합의 못 하면 별도 judge 에이전트가 최종 판단

MVP 단계에서는 비교 실험의 통제를 위해 **고정 라운드 방식**으로 시작하고, 이후 조기 종료·judge 방식을 추가 비교.

→ Learning Goal **(5) Debate 및 대기시간 최소화** 중 "multi-agent debate, debate structure (critic / judge)", "early stopping"

### 4.4 Hard Negative 테스트셋 구축 — 아첨(self-sycophancy) 측정 방법론

"critic이 안일하게 대충 넘기는가"를 측정하려면, critic이 틀린 곳을 잡아내는지 검증할 수 있는 정답이 있는 테스트셋이 필요함. 사용자의 영어 설명에 **의도적으로 그럴듯한 오개념을 주입**한 입력을 만들어, single-prompt critic과 debate critic 중 누가 더 잘 잡아내는지 비교. 이게 2.1(self-sycophancy 견제)을 실제로 정량 측정하는 방법.

→ Learning Goal **(4) 아첨, 환각 억제** 중 "critic이 틀린 곳을 탐지하도록 설계 (의도적으로 틀린 설명 주입해서 탐지 여부 확인)"

### 4.5 대기 시간(latency) 처리 — batch 우선, 최적화는 나중

실시간 streaming + multi-agent 호출은 느림. MVP는 "입력 종료 → 처리" batch 방식. 단, 3.1(self-critique)과 3.2(debate, R/N/K)를 같은 batch 파이프라인으로 비교 가능하게 만들어야 3.3 최종 목표(정확도 대비 속도·비용 최적점 탐색)가 성립함. streaming·continuous batching 같은 서빙 최적화는 Phase 3.

→ Learning Goal **(5) Debate 및 대기시간 최소화** 중 "KV Cache, Flash Attention, Paged Attention", "vLLM, GPU parallelism" (단, 이 단계에선 적용 안 하고 Phase 3로 미룸)

---

## 5. Eval Harness 설계 (연구 핵심)

위에서 정의한 변수·아키텍처·테스트셋을 하나의 실험 설계로 통합한다.

### 5.1 비교 대상

3.1(Architecture 1 — Self-Critique)과 3.2(Architecture 2 — Multi-agent Debate)를 **같은 입력셋**으로 돌려 비교하는 것이 이 harness의 존재 이유. "multi-agent" 안에 R/N/K 세 축이 있다는 게 핵심.

### 5.2 독립변수 (실험 조작 변인)

| 변수                | 정의                                                 | 실험 범위(초기) |
| ------------------- | ---------------------------------------------------- | --------------- |
| Architecture        | Self-Critique vs Multi-agent Debate                  | {1, 2}          |
| Debate round 수 (K) | 의견 충돌 시 토론 라운드 수 (4.3: MVP는 고정 라운드) | {0, 1, 2, 3}    |
| Ensemble 크기 (N)   | 같은 역할 critic 병렬 수                             | {1, 2, 3}       |
| 역할 다양성 (R)     | critic 역할 종류 수 (3.2.1: Conceptual, Procedural)  | {1, 2}          |

3.3에서 말한 "최적점 탐색"은 이 네 변수의 조합 공간(design space)에서 종속변수가 가장 좋은 지점을 찾는 일이다.

### 5.3 종속변수 (측정 지표)

- **평가 정확도** — 4.4의 Hard Negative 테스트셋 기준, critic이 의도적으로 주입된 오개념을 잡아내는 비율(recall) + 정상 설명을 오답으로 잘못 판정하는 비율(false positive)
- **응답 시간(latency)** — 입력부터 최종 Synthesizer 출력까지 걸린 시간
- **토큰 사용량(token usage)** — 전체 파이프라인이 소비한 입출력 토큰 합

### 5.4 데이터셋 — 두 종류 필요

1. **Hard Negative 테스트셋** (4.4) — 의도적으로 그럴듯한 오개념을 주입한 입력. critic의 탐지 능력(정확도) 측정용.
2. **정상 설명 데이터셋** — 사용자가 실제로 전공 내용을 영어로 설명한(또는 합성한) 정상 입력. False positive 측정 및 일반적인 latency/token 측정용.

두 데이터셋 다 강의자료 grounding 대상(4.1)과 짝이 맞아야 함 — Hard Negative를 만들 때도 "이 강의자료 기준으로 틀린 것"이어야 채점 가능.

### 5.5 재현성 (Reproducibility)

결과는 재현 가능하게 시드·설정을 기록한다. Architecture/K/N/R 조합별로 같은 입력 batch를 사용해야 비교가 공정함 — 입력 순서·샘플링 시드를 조합 간 고정.

### 5.6 ICICPE 페이퍼 연결 데이터 보존

debate 결과(critic들의 반박 과정, 합의에 이르는 토론 로그)는 ICICPE 페이퍼(debate-distilled fine-tuning) 아이디어와 연결될 수 있는 데이터이므로, eval 실행 시 최종 점수만 남기지 않고 **각 critic의 1차 평가, 라운드별 반박 내용, 최종 합의까지 전체 토론 로그를 별도로 저장**한다.

---

## 6. 로드맵

**Phase 0 — 루프 닫기**
전공 내용을 영어 텍스트로 입력 → Architecture 1(Self-Critique) 단일 파이프라인 → 텍스트 출력. CLI/노트북. "설명하면 뭐라도 피드백이 나온다"를 가장 먼저 증명. 이 단계에서 LLM 호출 추상화(1.2 Learning Goal (3))를 깔아둬서 이후 Architecture 2와 비교 가능한 구조로 시작.

| 단계 | 내용                                                                        | Learning Goal                                           |
| ---- | --------------------------------------------------------------------------- | ------------------------------------------------------- |
| 0-1  | 모델 선택 (Qwen2.5-7B/Llama-3.1-8B/Mistral-7B 중 택1) + Colab GPU 환경 세팅 | (0) GPU가 왜, 얼마나 필요한가 — vRAM vs CPU RAM / Colab |
| 0-2  | 최소 repo 골격 구성 (`pyproject.toml`, `venv`, Github)                      | (0) Docker, 로컬 venv + Github                          |
| 0-3  | LLM 호출 추상화 레이어 구현 (single-call ↔ multi-agent 갈아끼울 수 있게)    | (3) LLM 호출 추상화, prompt 설계                        |
| 0-4  | Self-Critique 파이프라인 (Architecture 1) 구현                              | (4) 아첨의 발생 원인 및 해결책 — self-critique, debate  |
| 0-5  | NoteWriter 최소 버전 구현                                                   | — (조립 단계, 신규 학습 항목 없음)                      |
| 0-6  | CLI에서 끝단까지 한 번 돌려서 "설명하면 피드백이 나온다" 확인               | — (통합 검증)                                           |

**Phase 1 — Multi-agent Debate 구축**
3.2 아키텍처 구현: Conceptual Critic + Procedural Critic (R=2) 분리, Orchestrator의 합의/충돌 판정(4.2) 구현, 고정 라운드 debate(4.3) 구현, structured output(Pydantic)으로 critic 간 통신. CLI/Gradio.

| 단계 | 내용                                                                            | Learning Goal                                                 |
| ---- | ------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 1-1  | LangGraph 기초 학습 + 그래프 스켈레톤 작성                                      | (3) LangGraph                                                 |
| 1-2  | structured output 스키마 정의 (critic 출력용 Pydantic 모델)                     | (3) structured output (Pydantic) — critic 간 통신 스키마 정의 |
| 1-3  | Conceptual Critic, Procedural Critic 분리 구현 (각자 prompt·RAG 검색 대상 분리) | (3) separation of concerns (context, language critic 분리)    |
| 1-4  | Orchestrator — 의견 충돌 감지·합의 판정 로직 구현                               | (3) orchestration pattern                                     |
| 1-5  | Debate Round 구현 (고정 라운드 K)                                               | (5) multi-agent debate, debate structure (critic / judge)     |
| 1-6  | Synthesizer 구현                                                                | — (조립 단계)                                                 |
| 1-7  | Architecture 2 전체 파이프라인 연결 + CLI/Gradio로 동작 확인                    | (3) shared variable / state 격리 — race condition             |

**Phase 1.5 — Eval Harness 구축 (연구 핵심)**
5번에서 설계한 비교 프레임워크 실제 구현. Hard Negative 테스트셋(4.4) 1차 구축, Architecture 1 vs 2 비교, K/N/R 첫 스윕(1.1 연구 진행 순서 1단계: ensemble 크기 단독 스윕부터).

| 단계  | 내용                                                    | Learning Goal                                   |
| ----- | ------------------------------------------------------- | ----------------------------------------------- |
| 1.5-1 | Hard Negative 테스트셋 1차 구축 (의도적 오개념 주입)    | (4) critic이 틀린 곳을 탐지하도록 설계          |
| 1.5-2 | 정상 설명 데이터셋 구축                                 | (4) 학습·평가용 데이터 및 데이터 합성           |
| 1.5-3 | 평가 정확도·latency·token 측정 코드 작성 (`metrics.py`) | (5) 추론 지연의 원인 찾기                       |
| 1.5-4 | `run_comparison.py` — Architecture 1 vs 2 비교 실행     | — (연구 목표 1.1의 실행, 핵심 산출물)           |
| 1.5-5 | Ensemble 크기(N) 단독 스윕 — 연구 진행 순서 1단계       | (3) orchestration pattern / (5) GPU parallelism |

**Phase 2 — Grounding 고도화 & 역할 확장**
Content Critic RAG를 본격적으로 확장 (Conceptual/Procedural critic별 검색 대상 분리, 4.1). **Completeness Critic 추가** (3.2.1에서 보류했던 항목 — 커리큘럼 수준 ground truth 인프라가 갖춰진 후 도입). K/N/R 스윕 2단계(역할 다양성 단독 스윕)·3단계(조합 탐색, 3.3 최종 목표) 진행.

| 단계 | 내용                                                              | Learning Goal                                   |
| ---- | ----------------------------------------------------------------- | ----------------------------------------------- |
| 2-1  | vector DB 구축, 강의자료 임베딩                                   | (6) RAG, embedding, grounding                   |
| 2-2  | critic별 검색 대상 분리 적용 (Conceptual vs Procedural 쿼리 분기) | (6) RAG, embedding, grounding                   |
| 2-3  | Completeness Critic 추가                                          | (3) separation of concerns                      |
| 2-4  | 역할 다양성(R) 단독 스윕 — 연구 진행 순서 2단계                   | (3) separation of concerns                      |
| 2-5  | N×R×K 조합 탐색 — 연구 진행 순서 3단계 (최종 목표)                | (3) orchestration pattern + (5) GPU parallelism |

**Phase 3 — 서빙 최적화 & 강화학습 적용**
로컬 모델 서빙 최적화 적용 — KV Cache, Flash Attention, vLLM, GPU parallelism (1.2 Learning Goal (5), 4.5에서 의도적으로 미뤄둔 항목). Unsloth 파인튜닝·사후학습·강화학습(1.2 Learning Goal (4))을 적용해 critic 자체를 self-sycophancy에 강하게 만드는 실험. ICICPE 페이퍼(debate-distilled fine-tuning)와 직접 연결되는 단계 — 5.6에서 보존해둔 토론 로그가 이 단계의 학습 데이터가 됨.

| 단계 | 내용                                  | Learning Goal                                        |
| ---- | ------------------------------------- | ---------------------------------------------------- |
| 3-1  | KV Cache, Flash Attention, vLLM 적용  | (5) KV Cache, Flash Attention, Paged Attention, vLLM |
| 3-2  | GPU parallelism 실험                  | (5) GPU parallelism                                  |
| 3-3  | Unsloth 파인튜닝 적용                 | (4) Unsloth 파인튜닝                                 |
| 3-4  | 사후학습·강화학습 실험                | (4) 사후학습(post-training), 강화학습(RL)            |
| 3-5  | ICICPE 페이퍼용 토론 로그 데이터 정리 | — (연구 출판 작업)                                   |

---

## 7. 기술 스택

API 호출 없이 로컬/from-scratch로 LLM을 직접 서빙·가속·파인튜닝하는 것이 전제.

| 영역           | 선택                                                                 | 비고                                                                                                         |
| -------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 컴퓨팅 환경    | Colab 무료 T4 → 필요 시 Colab Pro                                    | M4 맥북 MPS는 보조용(가벼운 inference 테스트), 파인튜닝·강화학습·N개 critic 동시 서빙은 Colab GPU가 메인     |
| 모델 로딩/서빙 | HuggingFace Transformers, vLLM                                       | Architecture 1/2 비교 실험에서 동일 모델을 여러 critic 인스턴스로 띄워야 하므로 vLLM의 멀티 요청 서빙이 핵심 |
| 가속/최적화    | CUDA, KV Cache, Flash Attention, Paged Attention, GPU parallelism    | Phase 3 항목 (4.5에서 미뤄둔 서빙 최적화)                                                                    |
| 파인튜닝/정렬  | Unsloth, 사후학습(post-training), 강화학습(RL)                       | Phase 3 항목 — critic의 self-sycophancy를 줄이는 방향으로 직접 학습                                          |
| Orchestration  | LangGraph                                                            | N개 critic 병렬 호출 + Orchestrator 합의판정 + debate round 제어를 그래프로 표현                             |
| RAG            | vector DB + 강의자료 (Conceptual/Procedural critic별 검색 대상 분리) | 5.4 데이터셋과 짝 맞춰 구축                                                                                  |
| 데이터 합성    | 합성 데이터 생성 (Hard Negative 테스트셋, 5.4)                       | LLM을 이용해 의도적 오개념 주입 데이터 생성                                                                  |
| 컨테이너/격리  | Docker, (필요 시) 쿠버네티스                                         | critic 인스턴스 격리·자원 분배 (3.2.2 ensemble 크기 N과 직결)                                                |
| 저장           | SQLite + markdown 노트                                               | 토론 로그(5.6) 포함해서 저장 — ICICPE 연결 데이터                                                            |
| 형상관리       | 로컬 `venv` + Github                                                 |                                                                                                              |
| UI             | Gradio·Streamlit → React                                             | 우선순위 낮음                                                                                                |

> 주 머신: MacBook Air M4 (가벼운 로컬 inference·코드 작성용) + Colab GPU (T4 → 필요 시 Pro, 학습·멀티 critic 서빙용)
> LLM은 API 호출 없이 HuggingFace + vLLM으로 직접 서빙. 파인튜닝은 Unsloth.

_(이 환경 블록은 CLAUDE.md "환경" 섹션에도 동일하게 반영 필요)_

---

## 8. 디렉토리 구조

```
fletcher/
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── fletcher/
│   ├── agents/
│   │   ├── content_critic/      # Conceptual Critic, Procedural Critic (3.2.1)
│   │   ├── orchestrator.py      # 합의/충돌 판정(4.2), debate round 트리거(4.3)
│   │   ├── synthesizer.py
│   │   └── note_writer.py
│   ├── architectures/
│   │   ├── self_critique.py     # Architecture 1 (3.1)
│   │   └── debate.py            # Architecture 2 (3.2) — R/N/K 설정 가능
│   ├── rag/
│   │   └── lecture_notes/       # critic별 검색 대상 분리 (4.1)
│   ├── finetuning/               # Phase 3 — Unsloth, post-training, RL
│   ├── serving/                  # Phase 3 — vLLM, KV Cache 등 서빙 최적화
│   └── pipeline.py
├── eval/
│   ├── datasets/
│   │   ├── hard_negative/        # 4.4 — 의도적 오개념 주입 데이터셋
│   │   └── normal/               # false positive·일반 latency 측정용
│   ├── sweep_configs/            # K/N/R 조합 실험 설정 (5.2)
│   ├── debate_logs/              # 5.6 — ICICPE 페이퍼 연결 데이터 보존
│   ├── metrics.py
│   └── run_comparison.py         # Architecture 1 vs 2 비교 (5.1)
└── ui/
```

---

## 9. 진행 현황 체크리스트

## 9. 진행 현황 체크리스트

- [x] 프로젝트 기획 정리 (피벗: 음성 → 텍스트, sycophancy 프레임 재정의)
- [x] 아키텍처 설계 (Self-Critique vs Multi-agent Debate, R/N/K 변수 정의)
- [ ] 0-1. 모델 선택 + Colab GPU 환경 세팅
- [ ] 0-2. 최소 repo 골격 구성
- [ ] 0-3. LLM 호출 추상화 레이어 구현
- [ ] 0-4. Self-Critique 파이프라인 구현
- [ ] 0-5. NoteWriter 최소 버전 구현
- [ ] 0-6. 끝단까지 동작 확인
- [ ] 1-1. LangGraph 기초 학습 + 스켈레톤
- [ ] 1-2. structured output 스키마 정의
- [ ] 1-3. Conceptual/Procedural Critic 구현
- [ ] 1-4. Orchestrator 합의/충돌 판정 구현
- [ ] 1-5. Debate Round 구현
- [ ] 1-6. Synthesizer 구현
- [ ] 1-7. Architecture 2 전체 연결·동작 확인
- [ ] 1.5-1. Hard Negative 테스트셋 구축
- [ ] 1.5-2. 정상 설명 데이터셋 구축
- [ ] 1.5-3. 측정 코드 작성
- [ ] 1.5-4. Architecture 1 vs 2 비교 실행
- [ ] 1.5-5. Ensemble 크기(N) 스윕
- [ ] 2-1. vector DB 구축
- [ ] 2-2. critic별 검색 대상 분리
- [ ] 2-3. Completeness Critic 추가
- [ ] 2-4. 역할 다양성(R) 스윕
- [ ] 2-5. N×R×K 조합 탐색
- [ ] 3-1. KV Cache, Flash Attention, vLLM 적용
- [ ] 3-2. GPU parallelism 실험
- [ ] 3-3. Unsloth 파인튜닝
- [ ] 3-4. 사후학습·강화학습 실험
- [ ] 3-5. ICICPE 토론 로그 정리
