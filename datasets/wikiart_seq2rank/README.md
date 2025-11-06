# WikiArt-Seq2Rank Dataset

## Task

Predicting artistic success based on sequences of artworks.

This dataset is formulated as a **Sequential Multiple-Instance Learning (SMIL)** problem, where each timestep corresponds to a **set of artworks** (unordered instances) created within the same period of an artist’s career, and the goal is to predict a **sequence-level ranking** representing the artist’s overall success.

## Data Format

Each artist has:
- **Sequence** — `{year: [artwork_path1, artwork_path2, ...], ...}`
- **Rankings** — Multiple ranking metrics (eBooks, NYT, Wikipedia, Google Trends, art market, etc.)
- **Target** — Overall ranking aggregated from multiple sources.

## Features

Pre-extracted visual features using **ResNet-34**:
- **Location:** `datasets/wikiart_seq2rank/wikiart_seq2rank_features_resnet34.pkl`
- **Format:** `{artwork_filename: numpy_array}`

## Data Splits

The dataset supports two splitting strategies:
1. **stratified_split**
2. **time_series_split**

## Training

```bash
# Train Set2Seq Transformer
python3 main.py \
    --dataset=wikiart_seq2rank \
    --model=Set2SeqTransformer \
    --set_model_name=DeepSets \
    --sequence_model_name=Transformer \
    --batch_size=16 \
    --epochs=100 \
    --ranking_type=overall_ranking \
    --wikiart_split=stratified_split

# Train baseline
python3 train_baselines.py \
    --data_path=datasets/wikiart_seq2rank/wikiart_seq2rank.pkl \
    --features_path=datasets/wikiart_seq2rank/wikiart_seq2rank_features_resnet34.pkl \
    --set_aggregate=mean \
    --model=xgb \
    --wikiart_split=stratified_split \
    --ranking=overall_ranking
```

## Evaluation Metrics

- **Kendall’s Tau** — Rank correlation  
- **MSE** — Mean squared error  
- **MAE** — Mean absolute error  

## Notes

- Only artists with at least 10 artworks are included.  
- Rankings are normalized using Min-Max scaling.  
- Sequences vary in length (different career spans).  
- Set sizes vary per timestep (different numbers of artworks per year).  

## Citation

Please cite the following when using this dataset and model:

```bibtex
@article{efthymiou2024set2seq,
  title={Set2Seq Transformer: Temporal and Positional-Aware Set Representations for Sequential Multiple-Instance Learning},
  author={Efthymiou, Athanasios and Rudinac, Stevan and Kackovic, Monika and Wijnberg, Nachoem and Worring, Marcel},
  journal={arXiv preprint arXiv:2408.03404},
  year={2024}
}
```