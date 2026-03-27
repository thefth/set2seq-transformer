# Mesogeos Dataset (Short-term Wildfire Danger Forecasting)

## Task

Short-term wildfire danger forecasting using the **Mesogeos** dataset introduced in:

> *"Mesogeos: A multi-purpose dataset for data-driven wildfire modeling in the Mediterranean"*
	
This task focuses on predicting near-future wildfire danger levels based on spatiotemporal environmental, meteorological, and human-activity features.

## Data Format

The dataset contains sequences of fire-related observations over time, where each timestep represents a set of spatial instances (e.g., grid cells or regions) with corresponding features and temporal metadata.

See the `mesogeos_swdf.pkl` and `mesogeos_swdf_features.pkl` files for preprocessed sequence and feature data.

## Features

Features include meteorological indicators (temperature, humidity, wind), vegetation indices, topography, and human-related variables aggregated by region and date.  
Each timestep corresponds to a set of such feature vectors.

## Training Example

```bash
python3 main.py \
    --dataset=mesogeos \
    --model=Set2SeqTransformer \
    --set_model_name=DeepSets \
    --sequence_model_name=Transformer \
    --temporal_embedding_type=timestamp_time2vec \
    --min_year=2006 \
    --max_year=2019 \
    --batch_size=32 \
    --epochs=50
```

## Temporal Embeddings

Temporal embeddings are essential for modeling the seasonal and annual variability in wildfire dynamics.

```bash
# Time2Vec embeddings (recommended)
--temporal_embedding_type=timestamp_time2vec \
--min_year=2006 \
--max_year=2019
```

## Evaluation Metrics

Evaluation metrics for this task include:
- Area Under the ROC Curve (AUC)
- F1-score (macro)
- Accuracy
- Precision and Recall

## Citation

Please cite the original Mesogeos paper:

```bibtex
@inproceedings{
kondylatos2023mesogeos,
title={Mesogeos: A multi-purpose dataset for data-driven wildfire modeling in the Mediterranean},
author={Spyros Kondylatos and Ioannis Prapas and Gustau Camps-Valls and Ioannis Papoutsis},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2023},
url={https://openreview.net/forum?id=VH1vxapUTs}
}
```

And cite our work if you use the Set2Seq Transformer for this dataset:

```bibtex
@article{efthymiou2026set2seq,
  title={Set2Seq Transformer: Temporal and Position-Aware Set Representations for Sequential Multiple-Instance Learning},
  author={Efthymiou, Athanasios and Rudinac, Stevan and Kackovic, Monika and Wijnberg, Nachoem and Worring, Marcel},
  journal={arXiv preprint arXiv:2408.03404},
  year={2026}
}
```
