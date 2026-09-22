# DCA-CC4 (adapted)

This is explicitly a detector-to-response adaptation of Sahani and Liu's DCA method, not PPO and not a native defender controller.

- Preserved: Blue-visible alert aggregation, deterministic DBSCAN with `minPts=3`, temporal sequencing, finite attack-path states, tabular value iteration with Dyna replay, and the attacker reward structure (impact + inverse path length + objective bonus). The learned path value participates in cluster/target ranking and the inferred attacker action is emitted in diagnostics.
- Adapted: power loss is a normalized mission/service-impact proxy. The inferred observable target is connected to A4 by a frozen mapping: no evidence -> no-op; low/single-channel evidence -> analyse; high-confidence removable multi-channel evidence -> remove; persistent/non-removable multi-channel evidence -> restore.
- Hidden compromise flags and Red sessions are rejected at the tokenizer boundary. Targets can only be host names present in observable evidence.
- The attack-path reward ranks inferred attacker paths only; it is not a defender learning reward.

Adaptation boundary: the paper's domain-specific power-grid abductive rules (frequency, voltage, topology and loss-of-load equations) cannot be reproduced from CC4 observations. CC4 uses a finite observable `(host, evidence-types)` state and learned path action/value instead. This is an explicit modality adaptation, not a claim that the original physical-grid abductive rule base is implemented.

Source paper SHA256: `173fa04a1675e98ce636c2a8b2eda9eae5e3c9e2347f5f19cb4aafe6708e3cb1`.

The first-version CC4 replay bridge was audited separately; it is not a paper
implementation and is not used by this baseline. See `PROVENANCE_AUDIT.md`.
