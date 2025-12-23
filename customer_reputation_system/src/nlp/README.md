# Feature 2: Text Analysis & Classification (NLP with MindSpore)

This module provides NLP-powered text analysis for customer reports, enabling automatic sentiment detection, urgency classification, and credibility scoring.

## Overview

The NLP module analyzes report text (title + description) to extract:
- **Sentiment**: positive, negative, or neutral
- **Urgency**: low, medium, high, or critical
- **Credibility Score**: 0-1 score indicating text quality and reliability

## Architecture

```
Text Input → Preprocessing → Tokenization → MindSpore Model → Classification Results
```

### Components

1. **TextPreprocessor** (`preprocessor.py`)
   - Text cleaning and normalization
   - Vocabulary building
   - Tokenization and sequence conversion
   - Basic feature extraction

2. **ReportTextClassifier** (`model.py`)
   - MindSpore-based neural network
   - Two variants:
     - `SimpleTextClassifier`: Fast MLP-based (default)
     - `ReportTextClassifier`: LSTM-based (more accurate, slower)
   - Multi-task learning: sentiment + urgency + credibility

3. **TextAnalyzer** (`text_analyzer.py`)
   - Main interface for text analysis
   - Model loading and inference
   - Fallback to rule-based analysis if model unavailable

4. **Training Script** (`train.py`)
   - Trains model on reports from database
   - Automatic label generation from report_type and rating
   - Saves checkpoint and vocabulary

## Usage

### Basic Usage

```python
from src.nlp.text_analyzer import TextAnalyzer

# Initialize analyzer
analyzer = TextAnalyzer()

# Analyze a report
analysis = analyzer.analyze_report(
    title="Unauthorized charge",
    description="I was charged twice without authorization."
)

print(analysis['sentiment'])  # 'negative'
print(analysis['urgency'])     # 'high'
print(analysis['credibility_score'])  # 0.75
```

### Integration with Report Handler

The NLP module is automatically integrated into `ReportHandler`:

```python
from src.ingestion.report_handler import ReportHandler

handler = ReportHandler(enable_nlp=True)  # NLP enabled by default

# Submit report (NLP analysis happens automatically)
result = handler.submit_report(report_data)

# Get report with NLP analysis
report_with_analysis = handler.get_report_with_analysis(report_id)

# Or get NLP analysis separately
nlp_analysis = handler.analyze_report_text(report_id)
```

## Training the Model

### Step 1: Ensure you have training data

The model needs at least 10 reports in the database. Run `main.py` to create sample reports:

```bash
cd customer_reputation_system_data
python main.py
```

### Step 2: Train the model

```bash
cd src/nlp
python train.py
```

This will:
- Load reports from database
- Build vocabulary
- Train the model
- Save checkpoint to `src/nlp/models/report_text_classifier.ckpt`
- Save vocabulary to `src/nlp/models/vocab.json`

### Step 3: Test inference

```bash
python test_inference.py
```

## Model Architecture

### SimpleTextClassifier (Default)

- **Embedding Layer**: Maps words to dense vectors
- **Mean Pooling**: Bag-of-words style aggregation
- **MLP Layers**: 128 → 64 hidden units
- **Output Heads**:
  - Sentiment: 3 classes (positive/negative/neutral)
  - Urgency: 4 levels (low/medium/high/critical)
  - Credibility: Regression (0-1)

### ReportTextClassifier (Advanced)

- **Embedding Layer**: Word embeddings
- **Bidirectional LSTM**: Sequence processing
- **MLP Layers**: Feature extraction
- **Output Heads**: Same as SimpleTextClassifier

## Configuration

Edit `src/nlp/config.py` to customize:

- `MAX_SEQUENCE_LENGTH`: Maximum tokens per text (default: 200)
- `VOCAB_SIZE`: Vocabulary size (default: 5000)
- `EMBEDDING_DIM`: Embedding dimension (default: 128)
- `HIDDEN_UNITS`: MLP layer sizes (default: (128, 64))
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Training epochs (default: 20)

## Fallback Behavior

If the model is not trained or unavailable, the system falls back to rule-based analysis:

- **Sentiment**: Based on negative word count
- **Urgency**: Based on urgency keyword count
- **Credibility**: Based on text quality metrics

This ensures the system always provides analysis results, even without a trained model.

## File Structure

```
src/nlp/
├── __init__.py           # Module exports
├── config.py             # Configuration
├── preprocessor.py       # Text preprocessing
├── model.py              # MindSpore models
├── text_analyzer.py      # Main analyzer interface
├── train.py              # Training script
├── test_inference.py     # Test script
├── README.md             # This file
└── models/               # Model artifacts
    ├── report_text_classifier.ckpt
    ├── vocab.json
    └── feature_stats.npy
```

## Next Steps

- **Feature 3**: Reporter Credibility System (will use NLP credibility scores)
- **Feature 4**: Merchant Reputation Scoring (will use sentiment analysis)
- **Feature 5**: Anti-Gaming & Fraud Detection (will use urgency detection)

## Notes

- The model uses MindSpore 2.3.0+ (as specified in requirements.txt)
- Training requires reports in the database (run `main.py` first)
- Model checkpoint and vocabulary are saved in `src/nlp/models/`
- NLP analysis is stored in report metadata JSON in the database

