# Selection v2: 300 fresh questions on unseen databases

## Frozen protocol

- Candidate generation: HKUST Qwen3.8-27B, one generation round, four existing paths; original TypeSafe JEV once per pool.
- 300 BIRD train questions from 30 databases unused by the earlier selector data, ten questions per database. Gold SQL was checked for executable protocol inclusion before the online phase, then withheld from all generation and judging prompts.
- Database-disjoint split: 180 questions / 18 databases for training, 60 / 6 for validation, 60 / 6 for test. The exact split, question IDs and manifest hash are in `results/new_selector/selection_v2_300/protocol/splits.json`.
- Seven-view JEV scores and full-result execution hashes were collected for every frozen pool before any new answer labels were computed.
- The DPC-inspired probe policy and 36/12/12 attempted-pool budget were fixed in `results/new_selector/selection_v2_300/protocol/probe_policy.json` before labels.

## Methods

1. **Raw JEV** ranks executable candidates by original candidate score.
2. **Seven-view rule** is the earlier hand-weighted min aggregation, retained as an ablation.
3. **Route-aware consensus** combines raw JEV with support from distinct generation paths. Same-result clustering is a feature, never a correctness label.
4. **Pairwise views, execution, and fusion rankers** learn bounded corrections to raw JEV on training databases. The loss compares correct versus executable incorrect candidates within the same question. Hyperparameters are chosen only on selection validation data; stop thresholds use separate validation databases.
5. **JEV plus constrained counterfactual probe** considers only a close-score or original-result-tied pair. An HKUST tester proposes a tiny synthetic SQLite database; a second SQL-hidden request expresses the question as restricted Python. A unique match may change Top-1 only if output arity is not ambiguous. Failed, non-distinguishing and inconclusive checks leave Top-1 unchanged. This is a limited DPC-inspired probe, not a reproduction of DPC.

The seven-view audit uses only training labels: false-low and false-high rates, dimension correlations, within-question pair concordance and leave-one-view-out Top-1 rescue/regression. The ranker keeps only views with training evidence of benefit. Result hashes cover the full returned result; ordered and unordered signatures are stored separately. SQL AST and question-alignment features, empty result, row/column count, nulls, cluster size and independent route support are available for the local ranker.

## Online collection and probe diagnostics

- Frozen raw pools: **300/300**.
- Seven-view evidence: **300/300**.
- Execution evidence: **300/300**. One query required a longer re-execution timeout after a transient validity mismatch.
- Counterfactual attempts: **60/60**; 42 non-distinguishing, 10 failed, 7 decisive, 1 inconclusive. Recorded successful requests: 58 calls, 17,170 input tokens and 12,431 output tokens. Failed-call token counts are unavailable, so these are lower bounds on actual probe cost.

## Offline comparison

All 300 questions were labelled only after online collection. The new splits contain 180/60/60 questions and 545/184/192 candidates. For the ranker, the 180 new training questions were combined with 192 older training questions, giving 372 training pools and 205 within-question mixed-label candidate pairs. Older validation data (47 questions, distinct databases) selected ranker hyperparameters. The new 60-question validation split alone fitted the stop thresholds. The new 60-question test split was evaluated once.

| Method | Validation Top-1 | Test covered Top-1 | Test all-question Top-1 | Test rescues / regressions vs raw JEV |
|---|---:|---:|---:|---:|
| Raw JEV | 33/37 | **35/38 (92.1%)** | **35/60 (58.3%)** | — |
| Old seven-view min rule | 34/37 | 35/38 | 35/60 | 0 / 0 |
| Route-aware result consensus | 33/37 | 35/38 | 35/60 | 0 / 0 |
| JEV plus constrained counterfactual probe | 33/37 | 35/38 | 35/60 | 0 / 0 |
| Pairwise views correction | 33/37 | 34/38 | 34/60 | 0 / 1 |
| Pairwise execution correction | 33/37 | 35/38 | 35/60 | 0 / 0 |
| Pairwise views plus execution | 33/37 | 35/38 | 35/60 | 0 / 0 |

The test has only three *covered* pools where raw JEV picks the wrong candidate: `legislator/4815`, `genes/2492`, and `genes/2494`. None of the additions rescues them. The 22 uncovered test pools cannot be fixed by selection alone. The result is a **negative ablation**: the implementation adds useful diagnostics and training records, but this frozen test does not show higher selection accuracy. With only three hard covered test pools, it also cannot rule out a benefit on a larger or different hard-pool sample.

### Stop thresholds

The prespecified validation rule chose the threshold accepting the most validation questions with empirical accepted-error fraction at most 10%. It did not use test labels. These are ranking-score cutoffs, not calibrated probabilities.

| Method | Validation threshold | Validation accepted / wrong | Test accepted / wrong |
|---|---:|---:|---:|
| Raw JEV | 0.9700 | 3 / 0 | 3 / 0 |
| Old rule | 0.8950 | 13 / 1 | 22 / 4 (18.2% wrong) |
| Consensus | 0.9775 | 3 / 0 | 3 / 0 |
| JEV plus probe | 0.9700 | 3 / 0 | 3 / 0 |
| Pairwise views | 1.0200 | 3 / 0 | 3 / 0 |
| Pairwise execution | 1.0096 | 7 / 0 | 7 / 1 (14.3% wrong) |
| Pairwise fusion | 1.0200 | 3 / 0 | 2 / 0 |

The rule and execution cutoffs violate the 10% target on unseen databases. Zero errors among two or three accepted questions is too small to establish a safe stop policy. Do not deploy any new direct-accept threshold based on this run.

### Audit and cost

The 372-pool training audit found 1,124 candidate instances, of which 689 matched the reference execution result. Raw JEV candidate AUC was 0.755 and within-question correct-versus-wrong pair concordance 0.754. The training-only screening retained `forward_entities`, `forward_conditions`, `reverse_predicates`, and `reverse_operations`; the other three views were removed from the learned correction. Even these retained views supplied no test rescue, and the views-only ranker regressed once. The former min rule remains in the report solely as a baseline; it is not the recommended selector.

Across all 300 new questions, successful raw traces recorded 2,144 generator calls, 5,090,125 input tokens and 826,003 output tokens. The first-round TypeSafe JEV recorded 293 calls, 968,055 input tokens and 43,241 output tokens. Seven-view scoring added another 293 recorded TypeSafe calls, 1,088,611 input tokens and 123,665 output tokens. The constrained probe attempted 60 pools (36/12/12 by split); successful request accounting contains 58 calls, 17,170 input tokens and 12,431 output tokens. Ten failed probes have unavailable request token counts. The initial TypeSafe pass also had 52 timed-out pool judgments before the successful retry, and their request cost is not in the successful traces. Thus all reported API costs are lower bounds on actual request cost. In the test split, the probe made 12 attempts but had **zero decisive** results; it changed no answer. Seven-view scoring added 58 recorded TypeSafe calls and 219,747 tokens to that split with no accuracy gain.

### Label quality and data reuse

Labels mean execution agreement with the BIRD reference on one database, including result multiplicity. They do not prove semantic correctness. For example, the reference query for `genes/2494` returns one gene's function even though the question asks about a pair of genes; this case needs human review before using it as a strong semantic training label. `genes/2492` also has an ambiguous relation between “most chromosomes” and the reference query's descending chromosome value. The records flag diagnosis as weak automatic data.

The versioned per-pool exports in `results/new_selector/selection_v2_300/training_export/` contain the question, full schema, SQL and route, execution summary and result-cluster features, raw JEV outputs, seven-view scores, candidate and pool labels, probe record, token usage and pool fingerprint. They are ready for **audited** JEV fine-tuning and future threshold fitting; the test export must remain held out from tuning. Detailed test decisions and the full risk–coverage curves are in `test_report.jsonl` and `fit/risk_coverage.json`.

### Conclusion

Keep raw JEV Top-1 as the current selector. The remaining bottleneck on this test is pool coverage (38/60), followed by three hard covered pools. The constrained DPC-inspired tester frequently produced non-distinguishing fixtures; a future version should target structural disagreement and validate fixture quality before paying for a Python solver. A larger independently adjudicated hard-pool set is needed before fitting or claiming a better stop rule.
