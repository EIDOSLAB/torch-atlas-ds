# Atlas Dataset
A map-style dataset format for PyTorch made for storing large amounts of data. You can read the format specification [here](docs/FORMAT.md).

## Example Workflows

### **Creating a Dataset**
```python
from atlas_dataset.writer import AtlasDatasetWriter

# Initialize the writer
with AtlasDatasetWriter("dataset_root", shard_size=1000, block_size=100) as writer:
    for example in examples:  # `examples` is your data source
        writer.add_example(example)
```

### **Reading a Dataset**
```python
from atlas_dataset.atlas import AtlasDataset

# Load the dataset
dataset = AtlasDataset("dataset_root")

# Access an example
example = dataset[42]
```

---