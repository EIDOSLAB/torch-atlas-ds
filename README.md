# Atlas Dataset
A map-style dataset format for PyTorch made for storing large amounts of data. You can read the format specification [here](docs/FORMAT.md).

## Features
- Map style (\_\_getitem\_\_ and \_\_len\_\_ methods to support random access)
- Sharding
- Compression (optional and with different supported strategies, more info [here](docs/FORMAT.md))
- Fast in-memory decompression (only a small section of a shard containing the requested index is decompressed)
- Fast random access and even faster sequential access
- Store examples in any format you want (uses pickle to serialize examples)

## Limitations
- Streaming is not supported to improve random access efficiency (the dataset must be stored on the machine / cluster you use for training)
- Currently, you cannot modify / append / delete examples in an existing Atlas Dataset (unless you create your own script), but I plan to improve this in the near future

## Installation

Currently the project is hosted only on github, to install it use:

```bash
pip install git+https://github.com/EIDOSLAB/torch-atlas-ds.git
```

or, if you use poetry:

```bash
poetry add git+https://github.com/EIDOSLAB/torch-atlas-ds.git
```

## Example Workflows

### **Creating a Dataset**
```python
from torch_atlas_ds import AtlasDatasetWriter

# Initialize the writer
with AtlasDatasetWriter("dataset_root", shard_size=1000, block_size=100) as writer:
    for example in examples:  # `examples` is your data source
        writer.add_example(example)
```

### **Reading a Dataset**
```python
from torch_atlas_ds import AtlasDataset

# Load the dataset
dataset = AtlasDataset("dataset_root")

# Access an example
example = dataset[42]
```

---

## Warnings

- Since the dataset uses pickle, only use datasets in atlas dataset format from trusted parties, since pickle may be used to execute unwanted code. If you are the author of the dataset, no need to worry about this.
- The format of the dataset may change in the future if the need arises since this project is still young (currently no change is planned)

## Author

Luca Molinaro, PhD Student @ UniTO
