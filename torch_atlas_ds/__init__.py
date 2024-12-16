"""
Atlas Dataset Package

This package provides tools for working with the Atlas dataset format, including
reading, writing, and managing datasets and their shards. The format is designed
to efficiently store large datasets with compression and is optimized for 
fast random access and reduced storage and memory requirements.

Modules:
    atlas:
        Contains classes for reading Atlas datasets and shards.
    writer:
        Provides classes for writing datasets and shards in the Atlas dataset format.

Exports:
    AtlasDataset: 
        A dataset reader for handling multi-shard Atlas datasets.
    AtlasDatasetShard: 
        A shard reader for handling individual shards of an Atlas dataset.
    CompressionStrategy: 
        Enum specifying available compression strategies.
    AtlasDatasetMetadata: 
        Named tuple containing metadata for an Atlas dataset.
    AtlasDatasetShardMetadata: 
        Named tuple containing metadata for an individual shard.
    AtlasDatasetWriter: 
        A writer for creating multi-shard Atlas datasets.
    AtlasDatasetShardWriter: 
        A writer for creating individual shards within an Atlas dataset.

Example:
    Reading an Atlas dataset:
        ```python
        from torch_atlas_ds import AtlasDataset

        dataset = AtlasDataset('/path/to/dataset')
        example = dataset[0]
        ```

    Writing an Atlas dataset:
        ```python
        from torch_atlas_ds import AtlasDatasetWriter

        with AtlasDatasetWriter('/path/to/output', shard_size=1000, block_size=100) as writer:
            for example in examples:
                writer.add_example(example)
        ```

Author:
    Luca Molinaro
"""

from .atlas import AtlasDataset, AtlasDatasetShard, CompressionStrategy, AtlasDatasetMetadata, AtlasDatasetShardMetadata
from .writer import AtlasDatasetWriter, AtlasDatasetShardWriter
