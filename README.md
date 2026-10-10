# Query History Retention Ablation — GPT-6 Luna

## 1. 실험 개요

**목적:** Multi-step Retrieval 과정에서 생성한 이전 Query Rewrite를 최종 LLM Reasoning에 유지하는 것이 Answer Accuracy에 도움이 되는지 평가한다.

### 실험 설정

| 항목 | 설정 |
|---|---|
| Dataset | HotpotQA Distractor |
| 평가 샘플 | 100개 (Bridge 50 / Comparison 50) |
| LLM | GPT-6 Luna |
| Retrieval | Dense Embedding |
| Retrieval Steps | 최대 2회 |
| Candidate Selection | LLM-based Document Selection |
| 평균 선택 문서 | 1.91개 |
| 평가 지표 | Answer EM, Token F1, Latency, Tokens |
| 통계 검증 | Paired Bootstrap 95% CI |

### 비교 조건

**A. Docs Only**
- 원본 질문과 최종 선택된 문서를 LLM에 전달
- 이전 검색 Query 내용은 제거
- 선택된 문서만 근거로 최종 답변 생성

**B. Query + Docs**
- 원본 질문과 동일한 선택 문서를 LLM에 전달
- 이전 검색 단계의 Rewrite Query도 함께 유지
- Tool Call History를 참고해 최종 답변 생성

두 조건에서 Retrieval 및 Document Selection 결과는 동일하게 고정한다.

---

## 2. 전체 실험 결과 (n=100)

| Metric | Docs Only | Query + Docs | Delta (B−A) |
|---|---:|---:|---:|
| **Answer EM** | **37.00%** | 34.00% | **−3.00%p** |
| **Answer F1** | **0.5537** | 0.5302 | **−0.0235** |
| Avg Latency | 4.22s | **1.75s** | −2.47s |
| Prompt Tokens | **397.89** | 436.83 | +38.94 |
| Completion Tokens | 69.90 | **66.43** | −3.47 |
| Empty Answers | 0 | 0 | 0 |
| Format Failures | **0** | 2 | +2 |
| Answers over 8 words | 25 | 27 | +2 |

### 주요 결과

- **Docs Only가 EM에서 3.0%p, F1에서 약 2.35%p 우세**했다.
- Query History를 추가하면 Prompt Token이 평균 **9.8% 증가**했다.
- Query + Docs의 평균 Latency는 약 58.5% 낮았다. 그러나 이 차이가 Query History 때문에 발생했는지는 확인되지 않았다.
- Query + Docs는 2개 사례에서 Output Format Failure가 발생했다.
- 두 조건 모두 Empty Answer와 Length-limit 종료는 없었다.

### Paired 비교

| Metric | B 승리 | B 패배 | 동률 |
|---|---:|---:|---:|
| Answer EM | 3 | 6 | 91 |
| Answer F1 | 9 | 18 | 73 |

Query History 추가로 개선된 사례보다 성능이 떨어진 사례가 더 많았다.

### 통계적 유의성

| Metric | Delta (B−A) | 95% Confidence Interval |
|---|---:|---|
| EM | −0.0300 | [−0.0900, +0.0300] |
| F1 | −0.0235 | [−0.0625, +0.0183] |

**두 신뢰구간 모두 0을 포함한다.**

따라서 Query History 추가로 성능이 통계적으로 유의하게 감소했다고 결론 내릴 수는 없다. 다만 이번 표본에서는 개선 근거가 관찰되지 않았다.

---

## 3. 질문 유형별 결과

### Bridge QA (n=50)

| Metric | Docs Only | Query + Docs | Delta |
|---|---:|---:|---:|
| **EM** | **0.56** | 0.50 | −0.06 |
| **F1** | **0.7195** | 0.7075 | −0.0120 |
| Latency | 1.92s | 1.77s | −0.15s |
| Prompt Tokens | 409.56 | 448.16 | +38.60 |

- EM: Query 유지 시 6%p 하락
- F1: 약 1.20%p 하락
- EM 승리/패배/동률: 1 / 4 / 45
- F1 승리/패배/동률: 4 / 6 / 40
- F1 Delta 95% CI: [−0.0601, +0.0443]

Bridge QA에서는 중간 검색 Query를 유지해도 Answer Quality가 개선되지 않았다.

### Comparison QA (n=50)

| Metric | Docs Only | Query + Docs | Delta |
|---|---:|---:|---:|
| **EM** | 0.18 | 0.18 | 0.00 |
| **F1** | **0.3879** | 0.3530 | −0.0349 |
| Latency | 6.52s | 1.73s | −4.79s |
| Prompt Tokens | 386.22 | 425.50 | +39.28 |

- EM: 동일
- F1: 약 3.49%p 하락
- EM 승리/패배/동률: 2 / 2 / 46
- F1 승리/패배/동률: 5 / 12 / 33
- F1 Delta 95% CI: [−0.0936, +0.0293]

Comparison QA에서는 Query History가 추가됐을 때 F1 하락이 상대적으로 크게 나타났다.

Comparison 유형의 낮은 EM과 긴 답변 비율을 고려하면, 정답 내용 오류와 불필요한 출력으로 인한 Exact Match 실패를 구분하는 추가 분석이 필요하다.

---

## 4. Retrieval 및 LLM Document Selection

| Metric | Result |
|---|---:|
| Retrieved Gold Support Recall | **1.000** |
| Selected Gold Support Recall | 0.915 |
| Selected All Support Rate | 0.840 |
| Average Selected Documents | 1.91 |
| Selector Fallback | **0** |
| First-step Empty Selection | 0 |
| Second-step Empty Selection | 91 |
| Second-step Skipped | 1 |
| Zero Selected Documents | 0 |

### 분석

**1. Dense Retrieval은 필요한 Gold Evidence를 모두 회수했다.**

Retrieved Gold Support Recall은 1.0으로, 제공된 Closed-pool Corpus에서 필요한 Gold Supporting Documents를 전부 검색했다.

**2. LLM Selection에서 일부 Gold Evidence가 제거되었다.**

- Selected Gold Support Recall: 91.5%
- 모든 Gold Supporting Documents를 보존한 문항: 84%
- 평균 선택 문서 수: 1.91개

이는 Candidate Reduction으로 Context 크기를 줄이는 과정에서 일부 필수 근거가 손실되었음을 의미한다.

**3. 두 번째 Retrieval에서 추가 문서를 거의 선택하지 않았다.**

- Second-step Empty Selection: 91/100
- Second-step Skipped: 1/100

두 번째 검색 단계에서 `selected_indices=[]`가 빈번하게 반환되었다.

가능한 원인은 다음과 같다.

- 첫 번째 Retrieval에서 충분한 정보 확보
- 두 번째 검색 결과가 기존 문서와 중복
- LLM Selector의 보수적인 문서 선택
- 추가 근거의 중요성을 Selector가 판단하지 못한 사례

Empty Selection이 정확한 판단이었는지 확인하려면 문항별 Gold Evidence Coverage를 분석해야 한다.

**4. Selector Fallback은 완전히 제거되었다.**

Fallback 0건으로, 이번 결과는 Dense Top-K로 자동 대체된 Selection 결과의 영향을 받지 않았다.

---

## 5. 성능 차이의 원인 분석

### Hypothesis 1: Rewrite Query의 정보 중복

최종 선택된 문서에 답변에 필요한 정보가 충분하다면 이전 Rewrite Query는 추가 근거를 제공하지 않을 수 있다.

이 경우 Query History는 Prompt Token을 증가시키지만 정답 정확도는 개선하지 않는다.

### Hypothesis 2: Query에 포함된 가정이 Reasoning에 영향

Rewrite Query는 검색 목적을 나타내며, 반드시 사실적으로 검증된 정보는 아니다.

검색 Query에 포함된 잘못된 가정이나 부정확한 Entity 관계가 최종 Reasoning에 영향을 줄 가능성이 있다.

이번 실험에서는 문항별 Query 오류 분석을 수행하지 않았으므로, 이는 가능한 원인에 해당한다.

### Hypothesis 3: Comparison QA의 출력 형식 문제

Comparison QA에서는 다음과 같은 현상이 관찰되었다.

- Docs Only: 50개 중 20개가 8단어 초과
- Query + Docs: 50개 중 21개가 8단어 초과
- 두 조건 모두 EM: 0.18

Comparison QA에서 긴 답변이 빈번하므로, 정답을 포함하면서 설명을 덧붙여 EM이 실패했을 가능성이 있다.

그러나 길이가 8단어를 초과했다는 사실만으로 오답의 원인을 판단할 수는 없다.

### Hypothesis 4: Latency 측정 변동

Query + Docs의 평균 Latency가 크게 낮았지만, Prompt Token은 오히려 더 많았다.

특히 Comparison QA에서 Docs Only의 평균 6.52초가 전체 차이에 크게 기여했다.

따라서 이번 Latency 차이는 실행 순서, API 서버 부하, Reasoning Token, 일부 지연이 큰 요청 등의 영향을 추가로 확인해야 한다.

Latency의 P50/P95와 교차 실행 순서(A→B, B→A)를 측정하기 전에는 Query History가 더 빠르다고 결론 내리지 않는다.

---

## 6. 종합 결론

### 핵심 발견

**HotpotQA 100개 평가에서 이전 Rewrite Query를 최종 Reasoning Context에 추가하는 전략은 Answer Accuracy를 개선하지 못했다.**

1. Docs Only의 EM/F1이 모두 높았다.
2. Query + Docs는 Prompt Token을 약 9.8% 더 사용했다.
3. 두 방식의 정확도 차이는 통계적으로 유의하지 않았다.
4. Dense Retrieval은 모든 Gold Evidence를 회수했으나, LLM Selection 이후 일부 근거가 손실되었다.
5. 두 번째 문서 선택에서는 Empty Selection이 91회 발생했다.

### 권장 Agent Architecture

```text
User Question + Selected History
              |
              v
       Query Rewrite
              |
              v
       Dense Retrieval
              |
              v
     LLM Document Selection
              |
              v
      Selected Documents
              |
              v
        Final Reasoning
              |
              v
          Answer
```

**기본 전략: Docs Only**

- 검색 Query는 Retrieval과 Tool Execution에 사용한다.
- 최종 Reasoning에는 원본 질문, 필요한 대화 History, 선택된 검색 근거를 전달한다.
- Query Trace는 별도로 보관하여 디버깅, 검색 오류 분석에 활용한다.
- Query History의 추가가 필요한 사례는 별도 평가로 결정한다.

### 실험의 한계

- HotpotQA는 본래 멀티턴 대화 데이터셋이 아니며, 이번 실험은 Multi-step Retrieval Query History에 대한 평가다.
- Question-local Closed-pool Retrieval 환경이므로 실제 대규모 검색 환경으로 일반화하기 어렵다.
- 모든 Gold Evidence가 최종 Selection을 통과한 것은 아니다.
- Query History가 올바른 경우와 잘못된 경우를 분리하지 않았다.
- 100개 표본에서 통계적으로 유의한 성능 차이가 확인되지 않았다.

### 최종 판단

**현재 설정에서는 Docs Only를 기본값으로 채택한다.**

Query History를 항상 Final Reasoning에 추가할 근거는 확인되지 않았다. 향후 실제 멀티턴 Agent 환경에서 History Selection, Tool Query Trace, Document Selection을 함께 평가해 이 결론을 검증할 필요가 있다.
