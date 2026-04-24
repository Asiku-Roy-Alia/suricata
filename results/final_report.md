# Hybrid IDS: Final Results Report

_Generated 2026-04-24 17:55:50_


## 1. Dataset Preparation

```
rows: 446632
columns: 79

Category distribution:
Category
BENIGN          379334
DoS              38751
DDoS             25603
Brute Force       1830
Web Attack         429
PortScan           391
Bot                287
Infiltration         7

Column list:
Flow Duration
Total Fwd Packets
Total Backward Packets
Total Length of Fwd Packets
Total Length of Bwd Packets
Fwd Packet Length Max
Fwd Packet Length Min
Fwd Packet Length Mean
Fwd Packet Length Std
Bwd Packet Length Max
Bwd Packet Length Min
Bwd Packet Length Mean
Bwd Packet Length Std
Flow Bytes/s
Flow Packets/s
Flow IAT Mean
Flow IAT Std
Flow IAT Max
Flow IAT Min
Fwd IAT Total
Fwd IAT Mean
Fwd IAT Std
Fwd IAT Max
Fwd IAT Min
Bwd IAT Total
Bwd IAT Mean
Bwd IAT Std
Bwd IAT Max
Bwd IAT Min
Fwd PSH Flags
Bwd PSH Flags
Fwd URG Flags
Bwd URG Flags
Fwd Header Length
Bwd Header Length
Fwd Packets/s
Bwd Packets/s
Min Packet Length
Max Packet Length
Packet Length Mean
Packet Length Std
Packet Length Variance
FIN Flag Count
SYN Flag Count
RST Flag Count
PSH Flag Count
ACK Flag Count
URG Flag Count
CWE Flag Count
ECE Flag Count
Down/Up Ratio
Average Packet Size
Avg Fwd Segment Size
Avg Bwd Segment Size
Fwd Header Length.1
Fwd Avg Bytes/Bulk
Fwd Avg Packets/Bulk
Fwd Avg Bulk Rate
Bwd Avg Bytes/Bulk
Bwd Avg Packets/Bulk
Bwd Avg Bulk Rate
Subflow Fwd Packets
Subflow Fwd Bytes
Subflow Bwd Packets
Subflow Bwd Bytes
Init_Win_bytes_forward
Init_Win_bytes_backward
act_data_pkt_fwd
min_seg_size_forward
Active Mean
Active Std
Active Max
Active Min
Idle Mean
Idle Std
Idle Max
Idle Min
Label
Category
```


## 2. Feature Pipeline

```
Input features: 77
RFE retained:   30
PCA components: 11
PCA variance retained: 0.9612

RFE-selected features:
Flow Duration
Total Fwd Packets
Total Length of Bwd Packets
Fwd Packet Length Max
Fwd Packet Length Std
Bwd Packet Length Max
Bwd Packet Length Min
Flow IAT Std
Flow IAT Max
Flow IAT Min
Fwd IAT Total
Fwd IAT Mean
Fwd IAT Std
Fwd IAT Max
Fwd IAT Min
Bwd Packets/s
Max Packet Length
Packet Length Mean
Packet Length Variance
Average Packet Size
Subflow Fwd Packets
Subflow Bwd Bytes
act_data_pkt_fwd
Active Mean
Active Std
Active Max
Idle Mean
Idle Std
Idle Max
Idle Min
```


## 3. Standard Held-Out Evaluation

### 3.1 Headline metrics

| model           |   macro_f1 |    mcc |   accuracy |   precision |   recall |   false_positive_rate |   false_negative_rate |   roc_auc |    ece |   tp |   fp |    tn |   fn |
|:----------------|-----------:|-------:|-----------:|------------:|---------:|----------------------:|----------------------:|----------:|-------:|-----:|-----:|------:|-----:|
| LinearSVC       |     0.9013 | 0.8134 |     0.9546 |      0.9607 |   0.7287 |                0.0053 |                0.2713 |    0.962  | 0.0255 | 9808 |  401 | 75466 | 3652 |
| IsolationForest |     0.7737 | 0.5482 |     0.8798 |      0.5924 |   0.6474 |                0.079  |                0.3526 |    0.8158 | 0.0852 | 8714 | 5995 | 69872 | 4746 |
| HybridStack     |     0.901  | 0.8122 |     0.9544 |      0.9566 |   0.7303 |                0.0059 |                0.2697 |    0.962  | 0.0085 | 9830 |  446 | 75421 | 3630 |


### 3.2 Per-category recall

| category     |   HybridStack |   IsolationForest |   LinearSVC |
|:-------------|--------------:|------------------:|------------:|
| BENIGN       |        0.9941 |            0.921  |      0.9947 |
| Bot          |        0      |            0.0345 |      0      |
| Brute Force  |        0      |            0      |      0      |
| DDoS         |        0.635  |            0.5118 |      0.635  |
| DoS          |        0.8486 |            0.7849 |      0.8458 |
| Infiltration |        0      |            1      |      0      |
| PortScan     |        0.0128 |            0.0897 |      0.0128 |
| Web Attack   |        0      |            0      |      0      |


### 3.3 Confusion matrices

```

LinearSVC
                pred_BENIGN  pred_ATTACK
true_BENIGN            75466         401
true_ATTACK             3652        9808

IsolationForest
                pred_BENIGN  pred_ATTACK
true_BENIGN            69872        5995
true_ATTACK             4746        8714

HybridStack
                pred_BENIGN  pred_ATTACK
true_BENIGN            75421         446
true_ATTACK             3630        9830

```


## 4. Leave-One-Attack-Category-Out (LOACO)

Each row below represents a full retraining run in which the named attack category was removed from the training set entirely. The *novel_recall* column measures the model's ability to detect that category at test time without ever having seen it during training.

| held_out_category   |   novel_recall |   known_attack_recall |   true_negative_rate |   overall_macro_f1 |   overall_mcc |   overall_fpr |
|:--------------------|---------------:|----------------------:|---------------------:|-------------------:|--------------:|--------------:|
| DoS                 |         0.7782 |                0.5636 |               0.9944 |             0.8849 |        0.7846 |        0.0056 |
| DDoS                |         0.635  |                0.788  |               0.9951 |             0.9024 |        0.8157 |        0.0049 |
| PortScan            |         0.0128 |                0.7336 |               0.9941 |             0.9007 |        0.8117 |        0.0059 |
| Brute Force         |         0      |                0.7541 |               0.9935 |             0.9012 |        0.8121 |        0.0065 |
| Web Attack          |         0      |                0.7345 |               0.9943 |             0.901  |        0.8125 |        0.0057 |
| Infiltration        |         0      |                0.7304 |               0.9942 |             0.901  |        0.8124 |        0.0058 |
| Bot                 |         0      |                0.7283 |               0.9933 |             0.8978 |        0.8058 |        0.0067 |


**Average novel-category recall:** 0.2037  

**Average recall on remaining known attacks:** 0.7189  

**Detection gap (known minus novel):** 0.5152


## 5. Plots

- **Reliability diagram (hybrid)**: `results\plots\reliability_hybrid.png`
- **Precision-recall curves**: `results\plots\pr_curves.png`
- **LOACO bar chart**: `results\loaco\loaco_plot.png`


## 6. Interpretation Notes


The standard held-out evaluation in Section 3 reports what the literature
typically calls benchmark performance. Numbers in that section should be
compared to the 2024 to 2025 peer-reviewed literature on CIC-IDS-2017, where
macro F1 above 0.95 is now routinely achieved by stacked ensembles.

The LOACO section is the more scientifically demanding test. A model that
matches benchmark accuracy but collapses on LOACO is not detecting novel
attacks, it is memorising attack signatures present in the training set. The
gap between known-attack recall and novel-category recall is the quantity to
discuss in the dissertation.

The reliability diagram in Section 5 shows whether the hybrid model's
probability outputs are trustworthy. A curve close to the diagonal indicates
that a predicted probability of 0.8 for an attack is empirically associated
with roughly 80% attack occurrence in that bin. Poor calibration undermines
any downstream decision threshold.
