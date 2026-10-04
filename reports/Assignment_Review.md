# Case Study Part 1: assignment review

Reviewed October 3, 2026 against the instructions supplied in chat. The separate Research Project Rubric was not supplied, so compliance with additional rubric criteria cannot be assessed.

The project covers every substantive step in the supplied assignment. The DOCX includes every required section, model comparison, per-class metrics, confusion matrices, ROC curves, business discussion, references and appendices. Important methodological limitations remain and are now disclosed rather than represented as validated deployment findings.

## Assignment coverage

| Requirement | Evidence | Assessment |
|---|---|---|
| Existing customer-feedback dataset with three sentiments | Root `Tweets.csv`, `src/data_loading.py`, paper Section 2 | Covered; airline/travel is an appropriate industry. No datasets added. |
| Dataset source, description and summary statistics | Notebook Section 1; paper Section 2; summary CSVs | Counts verified against the existing CSV. |
| Stop words, punctuation, tokenization and lemmatization | `src/preprocessing.py`, notebook Section 2, paper Section 3 | Covered; negation retention justified. Neural preprocessing variations explained. |
| Numerical features | TF-IDF word/character features and IDF-weighted Word2Vec | Covered; vectorizers/embeddings fit within training folds. |
| Models and rationale | NB, LR, two SVM configurations, Word2Vec LR, BiLSTM | Six evaluated configurations across four algorithm families. Transformer is implemented future work; it is not required by the assignment. |
| Hyperparameters and strategy | Code, saved model metadata, paper Section 5 | Selected parameters recorded; historical search provenance limitation disclosed. |
| Accuracy, precision, recall, F1 and model comparison | `model_comparison.csv`, `per_class_metrics.csv`, paper Section 6 | Recomputed from all six prediction files; verified. |
| Confusion matrix and/or ROC | 18 embedded paper figures, including both | Covered; document cross-references resolve. |
| Interpretation, limitations and conclusion | Paper Sections 7-8, notebook Section 6 | Covered; test-selected triage and convenience-sample limitations clarified. |
| APA or IEEE references | APA-style paper bibliography | Expanded abbreviated software-reference author lists using primary publication metadata. No claim of exhaustive DOI/reference validation. |
| Formal PDF or DOCX | `Sentiment_Analysis_Research_Paper.docx` | DOCX satisfies the specified alternative. Author is Albert Kabore, PhD Student in AI; course and instructor placeholders removed. |
| Documented Python/notebook and reproduction README | `src/`, `scripts/`, notebook, README | Covered. Fresh full training was not repeated during this review. |
| ZIP or GitHub link | Local repository | Submission ZIP is saved in reports/; upload it through your course system. |
| Case Study Part 2 presentation | Reminder included in README | Separate future assignment; not required for Part 1. |

## Verified results

The original CSV contains 14,640 rows. Cleaning removes 155 repeated tweet IDs, including 18 IDs with conflicting labels, leaving 14,485 tweets. Splits contain 10,139 training, 2,173 validation and 2,173 test tweets. All six prediction files contain the exact expected test IDs and labels; recalculated scalar metrics match the saved comparison within floating-point tolerance (maximum absolute error approximately 1.1e-16).

The saved word+character TF-IDF SVM leads by test macro-F1: accuracy **0.8141**, macro precision **0.7688**, macro recall **0.7632**, macro-F1 **0.7658**. BiLSTM macro-F1 is **0.7616**. Model ranking is descriptive test performance, not proof of statistically distinct macro-F1.

The audit is reproducible with `python scripts/audit_project.py`; evidence is saved to `results/metrics/research_audit.json`.

## Corrections and qualifications

1. Transformer results are absent. Removed claims of a completed Transformer comparison and unverified claims about why training failed. No additional model or data downloads were needed.
2. Added Holm-adjusted McNemar p-values. These tests assess paired error rates, not macro-F1. After adjustment, no error-rate difference is detected against the word-only SVM (p = 0.0583); the BiLSTM comparison remains significant at 0.05 (adjusted p = 0.0466). Nonsignificance does not prove equivalence; choosing a comparator from test ranking makes these comparisons exploratory.
3. Triage thresholds are chosen using test labels. Reported recall, precision and workload are exploratory operating points. Deployment thresholds require validation/out-of-fold selection followed by fresh evaluation.
4. Thirteen distinct lowercased texts occur in both test and development sets. IDs are disjoint, but independence is imperfect. Existing splits and verified predictions were retained; grouped/temporal evaluation is an improvement, not a completed result.
5. Classical models refit on 85% of data; neural models train on 70%. The comparison evaluates complete pipelines with unequal fitting budgets. Bootstrap intervals assume independent tweets and may understate uncertainty.
6. Saved LR TF-IDF metadata records 20 candidates versus 24 in current code; word+character SVM records 8 versus 10. Historical grids cannot be inferred from counts. The paper separates current definitions from saved selected parameters; future searches save exact grids.
7. Word2Vec reproducibility requires setting `PYTHONHASHSEED` before interpreter startup, along with seeds and fixed library versions. Setting it inside a running interpreter does not reseed hashing. The notebook warns about differing refit predictions.
8. Replaced unsupported annotation-unanimity language, untested probability-calibration claims, unmeasured Transformer token coverage, inaccurate family counts and accuracy claims based only on macro-F1. Corrected the balanced feature-space sampling description.
9. Dataset-wide Delta-to-JetBlue relabeling is an existing research assumption supported by prevalent handles, not verification of every tweet's carrier. Airline KPIs inherit this assumption. Strange strings in text are observable; their exact corruption mechanism is unverified.
10. Complaint reasons come from existing human annotations. Sentiment classification does not identify causal operational drivers or train a reason classifier. Population satisfaction, ROI, future workload and production readiness are not established by this sample.
11. Notebook neural retraining now persists its training metadata; notebook comparison output retains model metadata. Added checks that reject stale IDs, mismatched labels, duplicate IDs and nonfinite scores before comparison.

## Validation performed

- 33 automated tests passed, including new prediction-integrity and adjusted-significance checks.
- Saved split assignments, dataset cleaning counts, sentiment counts and all six models' scalar metrics audited against the existing CSV.
- Evaluation and insight stages rerun from existing predictions; research paper regenerated from corrected results.
- Every notebook code cell parsed successfully. Notebook execution and complete model retraining were not performed during this review.
- DOCX structure checked: required headings, 18 embedded figures and no unresolved figure/table reference markers. Rendered page layout was not visually inspected.
- Primary sources checked for DistilRoBERTa architecture and software-reference metadata: [model card](https://huggingface.co/distilbert/distilroberta-base), [scikit-learn publication](https://jmlr.org/papers/v12/pedregosa11a.html), [Transformers publication](https://aclanthology.org/2020.emnlp-demos.6/), [PyTorch publication](https://papers.nips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html).

## Before submission

Check the DOCX's rendered page layout in Word, and submit the DOCX plus the code ZIP or GitHub link. Apply any additional criteria in your separate Research Project Rubric. Historical tuning provenance and split dependence are disclosed limitations; a fresh complete experiment would be needed to resolve them.

## Writing and documentation review: October 3, 2026

The notebook now includes an abstract, an explicit dataset-source description, research objectives, preprocessing justifications, feature descriptions, model-selection rationale, and explanations of the evaluation metrics. Twenty-two observation sections remain beside the relevant outputs, with multiple findings and qualifications in each section. These sections distinguish measured results from interpretations and proposed improvements.

The research-paper generator was revised to remove grammatical errors, unsupported claims about the reason for missing Transformer results, causal interpretations of model coefficients, and premature operational claims. The introduction, objectives, discussion, conclusion, and reproducibility appendix now describe the completed experiment more precisely. The README distinguishes rerunning the pipeline from exact reconstruction of the saved historical searches.

The formal DOCX remains the Part 1 paper deliverable; the notebook is supporting code and documentation. The author is filled in, unwanted title-page placeholders are removed, and the submission ZIP is supplied. The separate Research Project Rubric was not provided. The Part 2 presentation is a subsequent deliverable.

## Final submission review: October 3, 2026

Rechecked the supplied assignment against the notebook, paper headings, source modules and saved results. All six assignment steps and all required paper sections are present. The separate Research Project Rubric was not supplied.

The final paper is `reports/Sentiment_Analysis_Research_Paper.docx`, with Albert Kabore as author and no course or instructor placeholders. Duplicate paper variants were removed. The submission ZIP contains this single paper and the current supporting codebase.

The data audit was rerun successfully: 14,640 source rows, 14,485 cleaned tweets, disjoint tweet IDs in the recorded splits, and six prediction files with matching source labels. All saved scalar comparison metrics were independently recomputed from those predictions and match within 1.2e-16. All 33 automated tests passed again. Notebook code cells parse successfully; no saved error outputs were found. Full notebook execution and fresh model training were not repeated, and rendered Word page layout was not visually inspected.

No synthetic customer feedback was added to the experiment. The notebook's hand-written preprocessing examples were replaced with four actual dataset records and their tweet IDs; their old outputs were cleared so they cannot be mistaken for output from the new code. These records come from the existing Kaggle-labelled airline dataset, not fabricated news. Sentiment annotations reflect annotator judgments and do not certify that every tweet's factual assertions are true. Agreement between predictions and reported metrics verifies internal consistency, not complete historical training provenance or dataset authenticity. Repeated-text overlap, unequal training budgets, missing historical grid definitions and exploratory test-selected thresholds remain disclosed.

Dataset source: [Twitter US Airline Sentiment on Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment). Part 2's presentation remains a separate assignment.
