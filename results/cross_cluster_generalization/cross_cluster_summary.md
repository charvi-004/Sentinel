# Cross-Cluster Generalization Experiment

Supplementary robustness experiment only; production evaluation and artifacts were not changed.

## Design

Leave-one-abuse-cluster-out evaluation across all four generated abuse clusters. Each held-out cluster is excluded entirely from training. Normal comparison rows are all normal transactions whose timestamps fall within the held-out cluster's inclusive timestamp range, selected deterministically without random sampling.

Holding out a cluster also holds out its abuse mechanism. This measures a combination of unseen-cluster and unseen-mechanism generalization, not proof of real-world generalization.

Features: 32 behavioral features from point-in-time feature generation; no labels, cluster IDs, or entity identifiers. StandardScaler and LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42) are fit only on each fold's training rows. Primary threshold: 0.50; no held-out-label threshold tuning. Simulated costs: FP=$5, FN=$100.

## Per-cluster results

| held_out_cluster | mechanism | positive_examples | normal_comparison_size | threshold | precision | recall | f1 | roc_auc | pr_auc | fpr | false_positives | false_negatives | simulated_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abuse_01 | shared_device | 45 | 8803 | 0.5000 | 0.0062 | 0.8444 | 0.0123 | 0.8049 | 0.0804 | 0.6940 | 6109 | 7 | 31245.0000 |
| abuse_02 | shared_payment_instrument | 45 | 9169 | 0.5000 | 0.0053 | 0.9333 | 0.0106 | 0.6585 | 0.0108 | 0.8522 | 7814 | 3 | 39370.0000 |
| abuse_03 | velocity_burst | 115 | 6505 | 0.5000 | 0.1110 | 0.8261 | 0.1957 | 0.9363 | 0.5171 | 0.1170 | 761 | 20 | 5805.0000 |
| abuse_04 | cross_merchant_ring | 150 | 3983 | 0.5000 | 0.2515 | 0.5533 | 0.3458 | 0.5946 | 0.4849 | 0.0620 | 247 | 67 | 7935.0000 |

## Aggregate results

Pooled across all held-out clusters and their deterministic normal comparison cohorts:

| Metric | Value |
| --- | --- |
| precision | 0.0170 |
| recall | 0.7268 |
| f1 | 0.0332 |
| fpr | 0.5246 |
| false_positives | 14931 |
| false_negatives | 97 |
| simulated_cost | 84355.0000 |
| roc_auc | 0.6663 |
| pr_auc | 0.1165 |

## Limitations

- Only four clusters exist, and each cluster is tied to one abuse mechanism.
- The 45-transaction clusters produce noisy estimates.
- Shared entities across generated activity may make the shift less independent than a true new-entity deployment case.
- The normal comparison is a time-window cohort, so fold sizes and class balance differ.
- Results are synthetic evaluation results and do not establish production robustness.
