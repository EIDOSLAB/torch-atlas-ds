"""
Atlas Dataset Reading Module

This module provides classes and utilities for reading datasets in the Atlas
format. It supports multi-shard datasets, enabling efficient access to large 
datasets stored across multiple files. The module also defines metadata classes 
and compression strategies for dataset configuration.

Classes:
    AtlasDataset:
        Handles multi-shard datasets, providing seamless access to data stored
        in multiple shards.
    AtlasDatasetShard:
        Manages individual shards within an Atlas dataset, enabling
        data retrieval and decompression.
    CompressionStrategy:
        Enum specifying the compression strategies supported by the Atlas 
        dataset format.
    AtlasDatasetMetadata:
        Contains metadata for an entire Atlas dataset, including shard sizes
        and compression strategy.
    AtlasDatasetShardMetadata:
        Stores metadata for individual dataset shards, such as block size, 
        stored examples, and compression configuration.

Example:
    Reading data from an Atlas dataset:
        ```python
        from torch_atlas_ds import AtlasDataset

        dataset = AtlasDataset('/path/to/dataset')
        example = dataset[42]
        ```

    Accessing shard-level data:
        ```python
        from torch_atlas_ds import AtlasDatasetShard

        shard = AtlasDatasetShard('/path/to/shard')
        example = shard[10]
        ```

Author:
    Luca Molinaro
"""

import zstandard
import pickle
import json
import numpy as np
import bisect
from enum import IntEnum
from itertools import accumulate
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, NamedTuple, List



class CompressionStrategy(IntEnum):
    """
    Enum for specifying the compression strategy used in the dataset.

    Attributes:
        NO_COMPRESSION (int): No compression applied to data.
        STANDARD_COMPRESSION (int): Zstandard compression without dictionary.
        SHARED_DICTIONARY_COMPRESSION (int): Zstandard compression using a shared dictionary across shards.
        DICTIONARY_COMPRESSION (int): Zstandard compression using an individual dictionary for each shard.
    """
    NO_COMPRESSION = 0
    STANDARD_COMPRESSION = 1
    SHARED_DICTIONARY_COMPRESSION = 2
    DICTIONARY_COMPRESSION = 3



class AtlasDatasetShardMetadata(NamedTuple):
    """
    Metadata for an individual shard of the Atlas dataset.

    Attributes:
        version (int): Version of the shard format.
        block_size (int): Number of examples per block in the shard.
        stored_examples (int): Total number of examples in the shard.
        compression_strategy (CompressionStrategy): Compression strategy used for the shard.
        compression_level (int): Compression level, see Zstandard supported compression levels.
        compression_dict_size (float): Size of the compression dictionary expressed as a fraction of the size of the uncompressed shard.
    """
    version: int
    block_size: int
    stored_examples: int
    compression_strategy: CompressionStrategy
    compression_level: int
    compression_dict_size: float



class AtlasDatasetMetadata(NamedTuple):
    """
    Metadata for the entire Atlas dataset.

    Attributes:
        version (int): Version of the dataset format.
        shard_sizes (List[int]): List of sizes (number of examples) for each shard.
        compression_strategy (CompressionStrategy): Compression strategy used across the dataset.
    """
    version: int
    shard_sizes: List[int]
    compression_strategy: CompressionStrategy



class AtlasDatasetShard(Dataset):
    """
    A PyTorch map-style dataset used for reading a single shard of the Atlas dataset.

    Attributes:
        VERSION (int): Supported version of the shard format.

    Methods:
        __init__(root: Path | str, mmap_index: bool = True, compression_dict_bytes: bytes | None = None):
            Initializes the shard object.

        __len__() -> int:
            Returns the total number of examples in the shard (relies on meta.json).

        __getitem__(index: int) -> Any:
            Retrieves an example by its index within the shard. If using a DataLoader, do not call this
            method manually before instantiating the DataLoader as this will initialize the zstandard
            decompressor object, which cannot be serialized with pickle, which is needed for the DataLoader
            to work.

    Parameters:
        root (Path | str): Path to the directory containing the shard data.
        mmap_index (bool): Whether to use memory-mapped access for the block index file (index.npy).
        compression_dict_bytes (bytes | None): Shared dictionary for decompression, if required.
    """
    VERSION = 1

    def __init__(self,
                 root: Path | str,
                 mmap_index: bool = True,
                 compression_dict_bytes: bytes | None = None
                 ) -> None:
        """
        Initializes an AtlasDatasetShard object.

        Parameters:
            root (Path | str): Path to the directory containing the shard data.
            mmap_index (bool): If True, enables memory-mapped access for the block index file (index.npy).
            compression_dict_bytes (bytes | None): Byte representation of a shared zstandard dictionary for decompression,
                                                required for SHARED_DICTIONARY_COMPRESSION strategy.
        """
        super().__init__()
        self.root = Path(root)
        self.mmap_index = mmap_index
        self.compression_dict_bytes = compression_dict_bytes
        self._initialized = False

        self.metadata = AtlasDatasetShardMetadata(**json.loads((self.root / 'meta.json').read_text()))

        if self.metadata.version != AtlasDatasetShard.VERSION:
            raise Exception('The Atlas Dataset version used in this shard is not supported')

        self.last_block = None
        self.last_block_idx = None
        self.decompressor = None
    
    def _lazy_init(self) -> None:
        if self._initialized:
            return

        self.block_index = np.load(self.root / 'index.npy', mmap_mode='r' if self.mmap_index else None)
        self.data_file = open(self.root / 'data.bin', 'rb')

        if self.metadata.compression_strategy == CompressionStrategy.DICTIONARY_COMPRESSION:
            comp_dict = zstandard.ZstdCompressionDict((self.root / 'zstd_dict.bin').read_bytes())
            self.decompressor = zstandard.ZstdDecompressor(dict_data=comp_dict)
        elif self.metadata.compression_strategy == CompressionStrategy.SHARED_DICTIONARY_COMPRESSION:
            if self.compression_dict_bytes is None:
                raise Exception('When using SHARED_DICTIONARY_COMPRESSION strategy you must provide a compression_dict')
            self.decompressor = zstandard.ZstdDecompressor(dict_data=zstandard.ZstdCompressionDict(self.compression_dict_bytes))
        elif self.metadata.compression_strategy == CompressionStrategy.STANDARD_COMPRESSION:
            self.decompressor = zstandard.ZstdDecompressor()
        
        self._initialized = True
    
    def __len__(self) -> int:
        """
        Returns the total number of examples in the shard (as stated in meta.json).

        Returns:
            int: Number of examples stored in this shard.
        """
        return self.metadata.stored_examples
    
    def __getitem__(self, index) -> Any:
        """
        Retrieves an example by its index within the shard.

        Parameters:
            index (int): Index of the example to retrieve. Supports negative indexing.

        Returns:
            Any: The example corresponding to the given index.

        Raises:
            IndexError: If the index is out of range.
        """
        if not self._initialized:
            self._lazy_init()

        if index < 0:
            index = len(self) + index
        
        if index >= len(self):
            raise IndexError('Index out of range')
    
        block_idx = index // self.metadata.block_size
        within_block_index = index % self.metadata.block_size

        if self.last_block_idx == block_idx and self.last_block is not None:
            return self.last_block[within_block_index]

        block_offset, next_block_offset = self.block_index[block_idx:block_idx+2]
        block_size = next_block_offset - block_offset

        self.data_file.seek(block_offset)
        block_bytes = self.data_file.read(block_size)

        if self.decompressor is not None:
            block_bytes = self.decompressor.decompress(block_bytes)
        
        block = pickle.loads(block_bytes)
        self.last_block_idx = block_idx
        self.last_block = block

        return block[within_block_index]



class AtlasDataset(Dataset):
    """
    A map-style PyTorch dataset in the Atlas dataset format, composed of multiple shards, with support for compression and fast random access.

    Attributes:
        VERSION (int): Supported version of the dataset format.

    Methods:
        __init__(root: Path | str, mmap_index: bool = True):
            Initializes the dataset object.

        __len__() -> int:
            Returns the total number of examples in the dataset.

        __getitem__(index: int) -> Any:
            Retrieves an example by its global index across all shards.

    Parameters:
        root (Path | str): Path to the root directory of the dataset.
        mmap_index (bool): Whether to use memory-mapped access for shard block index files (index.npy).
    """
    VERSION = 1

    def __init__(self,
                 root: Path | str,
                 mmap_index: bool = True
                 ) -> None:
        """
        Initializes an AtlasDataset object.

        Parameters:
            root (Path | str): Path to the root directory of the dataset.
            mmap_index (bool): If True, enables memory-mapped access for block index files of shards (index.npy).
        """
        self.root = Path(root)
        self.mmap_index = mmap_index

        self.metadata = AtlasDatasetMetadata(**json.loads((self.root / 'meta.json').read_text()))

        if self.metadata.version != AtlasDataset.VERSION:
            raise Exception('The version of this Atlas Dataset is not supported')

        shard_paths = sorted([p for p in self.root.iterdir() if p.is_dir()])

        shared_dict_bytes = None

        if self.metadata.compression_strategy == CompressionStrategy.SHARED_DICTIONARY_COMPRESSION:
            shared_dict_bytes = (self.root / 'zstd_dict.bin').read_bytes()

        self.shards = [
            AtlasDatasetShard(
                root=p,
                mmap_index=self.mmap_index,
                compression_dict_bytes=shared_dict_bytes
            )
            for p in shard_paths
        ]

        # This is useful when some shards are missing (multinode training with different shards used in different nodes)
        self.actual_shard_sizes = [len(shard) for shard in self.shards]
        self.cumulative_shard_sizes = list(accumulate(self.actual_shard_sizes))

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset.

        Returns:
            int: Number of examples across all shards.
        """
        return self.cumulative_shard_sizes[-1]
    
    def __getitem__(self, index) -> Any:
        """
        Retrieves an example by its global index across all shards.

        Parameters:
            index (int): Global index of the example to retrieve. Supports negative indexing.

        Returns:
            Any: The example corresponding to the given index.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0:
            index = len(self) + index
        
        if index >= len(self):
            raise IndexError('Index out of range')
        
        shard_idx = bisect.bisect_left(self.cumulative_shard_sizes, index + 1)

        if shard_idx == 0:
            within_shard_index = index
        else:
            within_shard_index = index - self.cumulative_shard_sizes[shard_idx - 1]

        return self.shards[shard_idx][within_shard_index]
