# Configuration Files

This directory contains YAML configuration files for training and evaluating models on the **Mesogeos** (short-term wildfire danger forecasting) and **WikiArt-Seq2Rank** (artist career prediction) datasets.

## Usage

### Basic
```bash
python3 main.py --config configs/mesogeos/set2seq_transformer.yaml
```

### Overriding Config Values
Command-line arguments override values in the config file:
```bash
python3 main.py --config configs/mesogeos/set2seq_transformer.yaml \
    --lr 0.001 \
    --epochs 50 \
    --batch_size 128
```

### Without a Config File
Run by specifying all parameters directly:
```bash
python3 main.py --dataset mesogeos --model Set2SeqTransformer \
    --data_path ../datasets/mesogeos/mesogeos_swdf.pkl \
    --features ../datasets/mesogeos/mesogeos_swdf_features.pkl \
    --setting 1
```

## Available Configurations

### Mesogeos (Short-term Wildfire Danger Forecasting)
- `mesogeos/set2seq_transformer.yaml` — Hierarchical Set-to-Sequence model (**recommended**)  
- `mesogeos/set_transformer.yaml` — Set Transformer (set encoder)  
- `mesogeos/transformer.yaml` — Standard Transformer  
- `mesogeos/lstm.yaml` — LSTM  
- `mesogeos/deepsets.yaml` — DeepSets  

### WikiArt-Seq2Rank (Artist Career Prediction)
- `wikiart_seq2rank/set2seq_transformer.yaml` — Hierarchical Set-to-Sequence model (**recommended**)  
- `wikiart_seq2rank/set_transformer.yaml` — Set Transformer (set encoder)  
- `wikiart_seq2rank/transformer.yaml` — Standard Transformer  
- `wikiart_seq2rank/lstm.yaml` — LSTM  
- `wikiart_seq2rank/deepsets.yaml` — DeepSets  

## Key Parameters

### Dataset
- `dataset`: Dataset name (`mesogeos` or `wikiart_seq2rank`)  
- `data_path`: Path to dataset pickle file  
- `features`: Path to feature file  
- `setting`: **[Mesogeos only]** timestep grouping factor  
- `wikiart_split`: **[WikiArt-Seq2Rank only]** split type (`stratified_split` or `time_series_split`)  
- `wikiart_ranking`: **[WikiArt-Seq2Rank only]** ranking type  

### Model
- `model`: Model architecture  
- `set_model_name`: Set encoder type (for `Set2SeqTransformer`)  
- `sequence_model_name`: Sequence encoder type (for `Set2SeqTransformer`)  
- `set_dim_hidden`: Hidden dimension of the set encoder  
- `set_output_dim`: Output dimension of the set encoder  
- `sequence_model_dim`: Hidden dimension of the sequence encoder  
- `sequence_num_heads`: Number of attention heads  
- `sequence_num_layers`: Number of Transformer/LSTM layers  

### Training
- `lr`: Learning rate  
- `batch_size`: Batch size  
- `epochs`: Maximum training epochs  
- `early_stopping_patience`: Early-stopping patience  
- `monitor_metric`: Metric to monitor (`auto`, `loss`, `accuracy`, `kendall_tau`)  
- `scheduler`: Learning-rate scheduler (`null`, `cosine`, `step`, `plateau`)  

### Embeddings
- `positional_embedding`: Positional embedding type  
- `temporal_embedding`: Temporal embedding type  
- `disable_temporal_embedding`: Disable temporal embeddings (for ablation)  

## Creating Custom Configs
1. Copy an existing config file.  
2. Modify parameters as needed.  
3. Save it with a descriptive name.  
4. Run:
   ```bash
   python3 main.py --config path/to/your_config.yaml
   ```

## Notes
- Paths in config files are relative to the directory where you run the script.  
- CLI arguments always override config file values.  
- The `auto` option for `monitor_metric` selects the appropriate metric for each task.  
- Embedding dimensions (`positional_embedding_dim`, `temporal_embedding_dim`) are inferred automatically.