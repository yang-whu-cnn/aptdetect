# AI_PROJECT_CONTEXT_REVIEW

## Purpose

This document records the current project understanding after reviewing:

- `AI_PROJECT_CONTEXT_APT_LLM_WM_RL.docx`
- `ug-cem-apt` branch structure
- `chapter2_region_detection` implementation organization

It is intended to help future AI sessions quickly restore project context.

---

## Repository Context

Repository:

```
yang-whu-cnn/aptdetect
```

Experiment branch:

```
ug-cem-apt
```

Main research directory:

```
chapter2_region_detection/
```

The branch is the implementation base for UG-CEM comparison experiments and the LLM + World Model + PPO framework.

---

## Final Research Framework

The final method is:

```
LLM Prior
    -> Candidate Response Plans
    -> World Model
    -> Future State + Uncertainty
    -> PPO Posterior Optimization
    -> Response Action
```

Design principles:

- LLM generates response priors, not direct actions.
- World Model predicts future environment evolution.
- PPO learns posterior policy correction from environment feedback.

---

## Frozen Action Space

High-level actions:

```
0: no-op
1: analyse
2: remove
3: restore
```

All comparison methods must use the same action contract.

---

## Experiment Comparison Design

Methods:

1. LLM + World Model + PPO (proposed)
2. UG-CEM / Uncertainty Guided Planning
3. Risk-sensitive Model-based RL
4. Knowledge-based Cyber Defence RL

Comparison principle:

Different algorithms may have different internal optimization objectives.
The final evaluation must use the same environment metrics.

Do not directly compare:

- PPO training reward
- CEM internal cost
- rule-based scores

Instead compare unified evaluation outputs:

- Attack Removal Time
- Host Work Fail
- Environment evaluation reward

---

## Current Repository Understanding

Important directories:

```
chapter2_region_detection/

├── baselines/
│   └── ug_cem_apt/
│
├── docs/
│
├── formal_experiments/
│
├── evaluation/
│
├── ours/
│
└── checkpoints/
```

`baselines/ug_cem_apt` is used for UG-CEM comparison implementation.

`ours/` contains proposed framework components including:

- LLM prior modules
- World Model runtime related modules
- PPO related modules

`evaluation/` contains gate tests, audits and experiment evaluation scripts.

---

## Current Implementation Status

Completed components:

- LLM interface
- Gemini/GPT model support
- World Model training
- B0.1 validation
- B2 multi-model validation
- B4 PPO pipeline
- CC4 action mapping
- B0.2 OOD audit

Current experimental route:

1. Complete multiple LLM model experiments.
2. Implement/verify baseline methods.
3. Complete LLM/WM/PPO ablation.
4. Generate paper experiment chapter.

---

## Important Fair Comparison Rules

When adding baselines:

Keep identical:

- environment
- observation/state definition
- action space
- reward evaluation
- seed protocol
- test scenarios

Only change the decision mechanism.

---

## Future Work Protocol

Before modifying core code:

```
Design confirmation
        ->
Documentation update
        ->
Implementation
        ->
Testing
        ->
Result recording
```

Avoid directly changing core algorithms without confirming experimental design.

---

## Current Review Status

Repository structure has been inspected.

Further detailed review should continue with:

1. `docs/`
2. `formal_experiments/`
3. `baselines/ug_cem_apt/`
4. `ours/`
5. `evaluation/`

Focus points:

- UG-CEM fairness
- reward/cost consistency
- World Model usage
- PPO posterior optimization implementation
- baseline reproducibility
