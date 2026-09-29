# VGEC 2026 EDA Report

## 1. Data Overview
- Total samples in V2: 457
- New samples in V2 (vs V1): 56

## 2. Key Findings
- Anomalous annotations found: 9
- Multi-reference samples: 56
- Unique duplicate groups: 450

## 3. Recommendations
- Review `annotation_anomalies.csv` to fix manual errors in gold data before training.
- Use `group_id` for splitting to avoid leakage.
- Pay attention to `edit_category` for curriculum learning or balancing.
