# mHealth Intake: Dual-Head BERT Pipeline for Mental Health Risk Assessment

## Overview
This repository contains the implementation of a mental health intake pipeline developed as part of an academic research study on early risk identification for depression and anxiety. The system analyzes self-reported textual responses and estimates independent risk signals for depression and anxiety using a shared deep language representation with task-specific output heads.
The work is intended for research and screening support purposes only and does not aim to provide clinical diagnosis or medical advice.
---

## Motivation
Mental health assessments often rely on static questionnaires and aggregate scoring methods that do not adequately capture nuanced linguistic cues or overlapping symptom profiles. Depression and anxiety frequently co-occur, yet their indicators may manifest differently in language. Treating them as a single outcome can obscure important distinctions.
This project explores a dual-output modeling strategy that allows shared semantic understanding while preserving task-specific decision boundaries for depression and anxiety risk estimation.
---

## Methodology
### Model Architecture
- A shared transformer encoder is used to learn contextualized representations of input text.
- Two independent classification heads are attached:
  - Depression risk head
  - Anxiety risk head
- Each head produces a probabilistic risk score using a sigmoid activation.
This design allows the model to capture common linguistic structure while avoiding forced coupling between the two outcomes.
### Representation Pooling
Standard max pooling and mean pooling were empirically evaluated and rejected due to sensitivity to noise and loss of discriminative signal.  
An 80th-percentile pooling strategy was adopted to retain salient linguistic cues while remaining robust to outliers.
### Loss Functions and Class Imbalance
- Binary Cross-Entropy loss is used as the base objective.
- Focal Loss is selectively applied to reduce false positives in anxiety detection.
- Class imbalance is handled through weighting and controlled sampling.
### Threshold Calibration
Instead of relying on a fixed 0.5 cutoff, decision thresholds are calibrated using validation-set F1 optimization. Separate thresholds are maintained for depression and anxiety to reflect differing prevalence and error trade-offs.
---

## Repository Structure
mhealth-intake/
├── app.py
├── train_two_heads.py
├── eval_bert_twoheads.py
├── model_infer.py
├── bert_probe.py
├── recalibrate_rc_thresholds.py
├── prep_clean.py
├── prep_split.py
├── prep_rc_split.py
├── configs/
├── prompts/
└── stats_lengths.py
Results
The dual-head architecture demonstrates improved interpretability and calibration compared to single-output baselines, particularly in scenarios where anxiety indicators are linguistically subtle or context-dependent. Threshold calibration further stabilizes predictions across varying response lengths and styles.
Detailed metrics and analysis are discussed in the associated research manuscript.

Ethical Considerations
This system is intended strictly for research, screening support, and exploratory analysis. It does not provide diagnoses and should not be used as a substitute for professional mental health evaluation.

Research Context
This implementation is directly linked to an academic research paper focused on multi-factor behavioral and psychological analysis using conversational and self-reported text. The repository serves as a reproducible reference for the modeling and evaluation methodology described in the study.

Future Work
Longitudinal modeling of intake responses
Uncertainty estimation and confidence calibration
Extension to additional behavioral dimensions
Fairness and bias audits across demographic groups

License
This project is released for academic and research use. Please cite the associated paper when referencing this work.
