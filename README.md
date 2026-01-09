# Forager

Enabling random access to large datasets on disk for PyTorch training and other use cases.

## Why Forager?

When training on large datasets (billions of tokens), you face a dilemma:

**Option 1: Load into memory**
- Works for small datasets
- Doesn't scale — a 1B token corpus needs gigabytes of RAM just for the text

**Option 2: Streaming / Iterable datasets**
- Scales to any size
- But: no true random shuffling (only buffer-based approximation)
- More complex handling: can't use `len(dataset)`, unclear epoch boundaries, custom resumption logic needed
- Can't use standard PyTorch `DataLoader(shuffle=True)`

**Why shuffling matters:** True random shuffling reduces gradient variance, prevents learning dataset ordering artifacts, and is especially important when mixing multiple data sources.

**Forager's solution:** Build a compact byte-offset index that enables O(1) random access to any sample via `seek()`. Your training code stays simple — large datasets work exactly like small ones:

```python
# Same code for 1K samples or 1B samples
dataset = JsonlDataset.create_from_index_on_filesystem('./data')
loader = DataLoader(dataset, batch_size=32, shuffle=True)  # True random shuffling!

for batch in loader:
    ...
```

No special iteration logic, no buffer management, no epoch hacks.

## Quick Start

### Use Case 1: Random Access to JSONL Files

```python
from data_forager.indexers.jsonl_indexer import create_default_jsonl_indexer
from data_forager.datasets.jsonl import JsonlDataset
from torch.utils.data import DataLoader

# One-time indexing (run once, reuse forever)
indexer = create_default_jsonl_indexer('./data')
indexer()
# Creates: ./data/index/file_location.txt, ./data/index/sample_locations.bin

# Training: random access with standard DataLoader
dataset = JsonlDataset.create_from_index_on_filesystem('./data')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    # batch is a list of dicts (parsed JSON objects)
    texts = [sample['text'] for sample in batch]
    ...
```

### Use Case 2: Tokenized Samples for Language Model Training

```python
from data_forager.indexers.tokenization_indexer import create_tokenize_and_index_jsonl_text_func
from data_forager.datasets.tokens import TokensDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

# One-time: tokenize JSONL text and create fixed-length samples
indexer = create_tokenize_and_index_jsonl_text_func(
    input_base_path='./corpus',
    tokenizer_func=tokenizer.encode,
    eos_idx=tokenizer.eos_token_id,
    sample_size=1024,  # Fixed context length
)
indexer()
# Creates: ./corpus/tokenized-samples/*.bin, ./corpus/index/*

# Training: fixed-length token sequences ready for NTP
dataset = TokensDataset.create_from_index_on_filesystem(
    './corpus',
    token_dtype=np.uint16,
)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in loader:
    # batch shape: (8, 1024) — ready for next-token prediction
    input_ids = batch[:, :-1]
    labels = batch[:, 1:]
    ...
```

## How It Works

Forager uses a two-phase approach:

### Phase 1: Indexing (One-Time)

Scan through your data files and record the byte offset of each sample:

```
sample_locations.bin:
┌─────────────┬─────────────┬───────────┐
│ file_index  │ byte_offset │ num_bytes │
│   uint64    │   uint64    │  uint64   │
├─────────────┼─────────────┼───────────┤
│     0       │      0      │    156    │  ← Sample 0: file 0, bytes 0-155
│     0       │    156      │    203    │  ← Sample 1: file 0, bytes 156-358
│     1       │      0      │    189    │  ← Sample 2: file 1, bytes 0-188
│    ...      │    ...      │    ...    │
└─────────────┴─────────────┴───────────┘
```

**Memory footprint:** 24 bytes per sample. A 1M sample dataset needs only ~24 MB for the index.

### Phase 2: Random Access (Training)

When you request `dataset[idx]`:
1. Look up `(file_index, byte_offset, num_bytes)` from the index
2. `seek()` to that position in the file
3. `read()` exactly `num_bytes`
4. Parse and return the sample

This is O(1) regardless of dataset size — no scanning, no loading everything into memory.

**Note:** Linux will cache frequently accessed data in the page cache when sufficient RAM is available, further improving performance.

## Components

### Index Stores

**`IndexStoreInterface`** — Protocol defining how indices are stored and loaded.

**`IndexStore`** (filesystem-based) — Default implementation storing indices as files:
- `file_location.txt` — List of data file paths
- `sample_locations.bin` — Binary array of (file_index, byte_offset, num_bytes) tuples

```python
from data_forager.index_stores.fs_based import IndexStore

# Used internally by indexers; rarely needed directly
store = IndexStore(base_path='./data', index_data_folder='index')
```

### Datasets

All datasets implement `__len__` and `__getitem__`, making them compatible with PyTorch DataLoader.

**`Dataset`** — Abstract base class providing:
- `create_from_index_on_filesystem(base_path)` — Load index and create dataset
- `initialize()` — Open file handles (called automatically on first access)
- Random access via `dataset[idx]` or `dataset[start:stop:step]`

**`JsonlDataset`** — Returns parsed JSON dicts:

```python
from data_forager.datasets.jsonl import JsonlDataset

dataset = JsonlDataset.create_from_index_on_filesystem('./data')
sample = dataset[0]  # Returns: {'text': '...', 'source': '...', ...}
```

**`TokensDataset`** — Returns numpy arrays of token IDs:

```python
from data_forager.datasets.tokens import TokensDataset
import numpy as np

dataset = TokensDataset.create_from_index_on_filesystem(
    './corpus',
    token_dtype=np.uint16,
)
sample = dataset[0]  # Returns: np.array([1534, 892, 2041, ...], dtype=uint16)
```

### Indexers

**`FileTextLinesIndexer`** — Base indexer for line-based text files. Scans files and records byte offsets for each line.

**`create_default_jsonl_indexer(input_base_path)`** — Creates an indexer for JSONL files:

```python
from data_forager.indexers.jsonl_indexer import create_default_jsonl_indexer

indexer = create_default_jsonl_indexer('./data')
indexer()  # Indexes all .jsonl files recursively
```

**`create_tokenize_and_index_jsonl_text_func(...)`** — Creates an indexer that:
1. Reads JSONL files
2. Extracts text (default: `sample['text']`)
3. Tokenizes using your tokenizer
4. Packs into fixed-length samples (with EOS separation)
5. Stores as binary files and builds index

```python
from data_forager.indexers.tokenization_indexer import create_tokenize_and_index_jsonl_text_func

indexer = create_tokenize_and_index_jsonl_text_func(
    input_base_path='./corpus',
    tokenizer_func=tokenizer.encode,  # Your tokenizer
    eos_idx=tokenizer.eos_token_id,  # EOS token ID
    sample_size=1024,  # Fixed context length (None for variable)
    token_dtype=np.uint16,  # Token storage dtype
)
indexer()
```

## Installation

```bash
pip install data-forager
```

Or install from source:
```bash
git clone https://github.com/visionscaper/data-forager.git
cd data-forager
pip install -e .
```

## Requirements

- Python >= 3.9
- numpy
- tqdm
- basics (visionscaper-pybase)

## License

MIT License
