"""
Atlas Dataset Writing Module

This module provides tools for creating datasets in the Atlas format. It 
supports writing data into block-based shards, with optional compression to 
optimize storage. Compression strategies include dictionary-based compression 
and shared dictionaries across multiple shards.

Classes:
    AtlasDatasetWriter:
        Manages the creation of multi-shard Atlas datasets, organizing data into 
        configurable shard and block sizes.
    AtlasDatasetShardWriter:
        Handles the creation of individual dataset shards, allowing data to 
        be added incrementally and compressed based on the chosen strategy.

Example:
    Writing a multi-shard dataset:
        ```python
        from torch_atlas_ds import AtlasDatasetWriter

        with AtlasDatasetWriter('/path/to/output', shard_size=1000, block_size=100) as writer:
            for example in examples:
                writer.add_example(example)
        ```

    Writing a single shard:
        ```python
        from torch_atlas_ds import AtlasDatasetShardWriter

        with AtlasDatasetShardWriter('/path/to/shard', block_size=100) as shard_writer:
            for example in examples:
                shard_writer.add_example(example)
        ```

Author:
    Luca Molinaro
"""

import zstandard
import pickle
import json
import bisect
import numpy as np
from pathlib import Path
from typing import Any
from torch_atlas_ds.atlas import AtlasDatasetShardMetadata, CompressionStrategy



class AtlasDatasetShardWriter:
    """
    Writer for creating a shard in the Atlas dataset format.

    Attributes:
        VERSION (int): The version of the shard format supported by the writer.

    Methods:
        add_example(example: Any) -> None:
            Adds an example to the shard. When using DICTIONARY_COMPRESSION strategy
            all examples will be kept into main memory until the writer is closed
            (this is needed to train the dictionary).

        close() -> None:
            Finalizes the shard, always call this method after all examples are added
            to the shard. Not calling this will result in an incomplete shard. This method
            will be called automatically if using this writer object as a context manager.

        __len__() -> int:
            Returns the number of examples currently stored in the shard.

        __enter__() -> AtlasDatasetShardWriter:
            Enables use of the shard writer as a context manager.

        __exit__(exc_type, exc_value, traceback) -> None:
            Finalizes the shard when exiting the context manager (calls the close method).
    """
    VERSION = 1

    def __init__(self,
                 path: Path | str,
                 block_size: int,
                 compression_strategy: CompressionStrategy = CompressionStrategy.DICTIONARY_COMPRESSION,
                 compression_level: int = 3,
                 compression_dict: zstandard.ZstdCompressionDict | None = None,
                 compression_dict_size: float = 0.01
                 ) -> None:
        """
        Initializes a shard writer for the Atlas dataset.

        Parameters:
            path (Path | str): Directory where the shard files will be written.
            block_size (int): Number of examples per block in the shard.
            compression_strategy (CompressionStrategy): Compression strategy to use for the shard.
            compression_level (int): Compression level for Zstandard compression.
            compression_dict (zstandard.ZstdCompressionDict | None): Compression dictionary for shared dictionary compression.
            compression_dict_size (float): Fraction of total uncompressed size of the shard to allocate for the compression dictionary.
        
        Raises:
            AssertionError: If `block_size` is not greater than 0.
            AssertionError: If `compression_dict_size` is not in the range [0.0, 1.0].
            Exception: If `compression_strategy` is SHARED_DICTIONARY_COMPRESSION but no `compression_dict` is provided.
        """
        self.path = Path(path)

        assert block_size > 0, 'block_size must be > 0'
        assert compression_dict_size >= 0.0 and compression_dict_size <= 1.0, 'compression_dict_size must be between 0 and 1'

        if compression_strategy == CompressionStrategy.SHARED_DICTIONARY_COMPRESSION:
            if compression_dict is None:
                raise Exception('When using SHARED_DICTIONARY_COMPRESSION strategy you must provide a compression_dict')
        
        self.metadata = AtlasDatasetShardMetadata(
            version=AtlasDatasetShardWriter.VERSION,
            block_size=block_size,
            stored_examples=0,
            compression_strategy=compression_strategy,
            compression_level=compression_level,
            compression_dict_size=compression_dict_size
        )

        self.path.mkdir()

        self.stored_examples = 0
        self.index = []
        self.shard_size_bytes = 0
        self.current_block = []
        self.pickled_blocks = []
        self.data_file = None
        self.compressor = None

        if compression_strategy == CompressionStrategy.SHARED_DICTIONARY_COMPRESSION:
            self.compressor = zstandard.ZstdCompressor(dict_data=compression_dict, level=compression_level)
        elif compression_strategy == CompressionStrategy.STANDARD_COMPRESSION:
            self.compressor = zstandard.ZstdCompressor(level=compression_level)

    def add_example(self, example: Any) -> None:
        """
        Adds an example to the shard.

        Parameters:
            example (Any): The example to add to the shard.
        """
        self.current_block.append(example)
        self.stored_examples += 1

        if len(self.current_block) == self.metadata.block_size:
            self._add_block()
    
    def _add_block(self) -> None:
        pickled_block = pickle.dumps(self.current_block)
        self.current_block = []

        if self.metadata.compression_strategy == CompressionStrategy.DICTIONARY_COMPRESSION:
            self.pickled_blocks.append(pickled_block)

        elif self.metadata.compression_strategy != CompressionStrategy.NO_COMPRESSION:
            assert self.compressor is not None
            self._write_block(self.compressor.compress(pickled_block))

        else:
            self._write_block(pickled_block)
    
    def _write_block(self, block: bytes) -> None:
        self.index.append(self.shard_size_bytes)
        self.shard_size_bytes += len(block)

        if self.data_file is None:
            self.data_file = open(self.path / 'data.bin', 'wb')
        
        self.data_file.write(block)
    
    def _update_metadata(self, key: str, value: Any) -> None:
        meta_dict = self.metadata._asdict()
        assert key in meta_dict
        meta_dict[key] = value
        self.metadata = AtlasDatasetShardMetadata(**meta_dict)

    def _save_index(self) -> None:
        max_val = self.index[-1]
        unsigned_limits = [
            np.iinfo(np.uint8).max,
            np.iinfo(np.uint16).max,
            np.iinfo(np.uint32).max,
            np.iinfo(np.uint64).max,
        ]
        unsigned_types = [np.uint8, np.uint16, np.uint32, np.uint64]
        index = bisect.bisect_left(unsigned_limits, max_val)

        if index == len(unsigned_limits):
            raise Exception(f"This shard is too big, maximum size in bytes must be less than 2^64")

        dtype = unsigned_types[index]
    
        index = np.array(self.index, dtype=dtype)

        np.save(self.path / 'index.npy', index, allow_pickle=False)

    def close(self) -> None:
        """
        Finalizes the shard. Writes any remaining examples, trains and saves compression dictionaries (if required),
        writes metadata, and closes the data file.

        Raises:
            Exception: If the shard size exceeds the maximum allowed size (2^64 bytes).
        """
        self._update_metadata('stored_examples', self.stored_examples)

        if len(self.current_block) > 0:
            self._add_block()
    
        if self.metadata.compression_strategy == CompressionStrategy.DICTIONARY_COMPRESSION:
            if len(self.pickled_blocks) >= 7: # if less than 7, it crashes because there are not enough samples to train the dictionary
                uncompressed_content_size = sum(len(x) for x in self.pickled_blocks)
                compression_dict_size = int(self.metadata.compression_dict_size * uncompressed_content_size) + 1
                compression_dict_size = max(compression_dict_size, 1024)
                compression_dict = zstandard.train_dictionary(compression_dict_size, self.pickled_blocks, level=self.metadata.compression_level)
                
                cctx = zstandard.ZstdCompressor(dict_data=compression_dict, level=self.metadata.compression_level)
                
                (self.path / 'zstd_dict.bin').write_bytes(compression_dict.as_bytes())
            else:
                self._update_metadata('compression_strategy', CompressionStrategy.STANDARD_COMPRESSION)
                cctx = zstandard.ZstdCompressor(level=self.metadata.compression_level)

            for block in self.pickled_blocks:
                compressed = cctx.compress(block)
                self._write_block(compressed)
            
            del cctx
    
        if self.data_file is not None:
            self.data_file.close()
        
        # add the size of the whole shard in bytes as last element of the index
        # makes it easier to calculate the size of the last block
        self.index.append(self.shard_size_bytes)
        
        self._save_index()
        
        (self.path / 'meta.json').write_text(json.dumps(self.metadata._asdict(), indent=4))

    def __enter__(self):
        """
        Enables the shard writer to be used as a context manager.

        Returns:
            AtlasDatasetShardWriter: The current shard writer instance.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Finalizes the shard when exiting the context manager.

        Parameters:
            exc_type: Exception type, if any occurred during the context.
            exc_value: Exception value, if any occurred during the context.
            traceback: Traceback object, if an exception occurred during the context.
        """
        self.close()
    
    def __len__(self) -> int:
        """
        Returns the number of examples currently stored in the shard.

        Returns:
            int: Number of stored examples.
        """
        return self.stored_examples



class AtlasDatasetWriter:
    """
    Writer for creating an Atlas dataset composed by multiple shards.

    Attributes:
        VERSION (int): The version of the dataset format supported by the writer.

    Methods:
        add_example(example: Any) -> None:
            Adds an example to the current shard. Creates a new shard if the shard size limit is reached.

        close() -> None:
            Finalizes the dataset, writing metadata and renaming shard directories.

        __enter__() -> AtlasDatasetWriter:
            Enables use of the dataset writer as a context manager.

        __exit__(exc_type, exc_value, traceback) -> None:
            Finalizes the dataset when exiting the context manager.
    """
    VERSION = 1

    def __init__(self,
                 path: Path | str,
                 shard_size: int,
                 block_size: int,
                 compression_strategy: CompressionStrategy = CompressionStrategy.SHARED_DICTIONARY_COMPRESSION,
                 compression_level: int = 3,
                 compression_dict_size: float = 0.01
                 ) -> None:
        """
        Initializes a dataset writer for the Atlas dataset.

        Parameters:
            path (Path | str): Directory where the dataset will be written.
            shard_size (int): Maximum number of examples per shard.
            block_size (int): Number of examples per block within each shard.
            compression_strategy (CompressionStrategy): Compression strategy to use for shards.
            compression_level (int): Compression level for Zstandard compression.
            compression_dict_size (float): Fraction of total uncompressed size of a shard to allocate for the compression dictionary.
                                            For shared dictionaries, this is relative to the uncompressed size of the first shard.
        """
        self.path = Path(path)
        self.shard_size = shard_size
        self.block_size = block_size
        self.compression_strategy = compression_strategy
        self.compression_level = compression_level
        self.compression_dict_size = compression_dict_size

        self.path.mkdir()

        first_shard_comp_strat = self.compression_strategy

        if self.compression_strategy == CompressionStrategy.SHARED_DICTIONARY_COMPRESSION:
            first_shard_comp_strat = CompressionStrategy.DICTIONARY_COMPRESSION

        self.current_shard = AtlasDatasetShardWriter(
            path=self.path / '0',
            block_size=block_size,
            compression_strategy=first_shard_comp_strat,
            compression_level=compression_level,
            compression_dict_size=compression_dict_size
        )
        self.shard_sizes = []
        self.shared_dict = None
    
    def _handle_shared_dict(self) -> None:
        if len(self.shard_sizes) == 1 and self.compression_strategy == CompressionStrategy.SHARED_DICTIONARY_COMPRESSION:
            shard_meta_path = self.current_shard.path / 'meta.json'
            shard_meta = json.loads(shard_meta_path.read_text())

            if shard_meta['compression_strategy'] == CompressionStrategy.STANDARD_COMPRESSION:
                # This happens when the first shard has less than 7 blocks
                self.compression_strategy = CompressionStrategy.STANDARD_COMPRESSION
            else:
                dict_path = self.current_shard.path / 'zstd_dict.bin'
                dict_bytes = dict_path.read_bytes()
                dict_path.rename(self.path / 'zstd_dict.bin')
                self.shared_dict = zstandard.ZstdCompressionDict(dict_bytes)
                shard_meta['compression_strategy'] = self.compression_strategy
                shard_meta_path.write_text(json.dumps(shard_meta, indent=4))
    
    def add_example(self, example: Any) -> None:
        """
        Adds an example to the current shard. If the current shard reaches the shard size limit, 
        finalizes the current shard and starts a new shard.

        Parameters:
            example (Any): The example to add to the dataset.
        """
        current_len = len(self.current_shard)

        if current_len < self.shard_size:
            self.current_shard.add_example(example)
            return
    
        self.current_shard.close()
        self.shard_sizes.append(current_len)

        self._handle_shared_dict()
        
        self.current_shard = AtlasDatasetShardWriter(
            path=self.path / str(len(self.shard_sizes)),
            block_size=self.block_size,
            compression_strategy=self.compression_strategy,
            compression_level=self.compression_level,
            compression_dict=self.shared_dict,
            compression_dict_size=self.compression_dict_size
        )

        self.current_shard.add_example(example)

    def close(self) -> None:
        """
        Finalizes the dataset. Writes metadata, renames shard directories, and closes the current shard.
        """
        self.shard_sizes.append(len(self.current_shard))
        self.current_shard.close()
        self._handle_shared_dict()
        
        meta = {
            'version': AtlasDatasetWriter.VERSION,
            'shard_sizes': self.shard_sizes,
            'compression_strategy': self.compression_strategy
        }

        (self.path / 'meta.json').write_text(json.dumps(meta, indent=4))
        
        num_digits = len(self.current_shard.path.name)

        for p in self.path.iterdir():
            if p.is_dir():
                p.rename(p.parent / f'{p.name:0>{num_digits}}')

    def __enter__(self):
        """
        Enables the dataset writer to be used as a context manager.

        Returns:
            AtlasDatasetWriter: The current dataset writer instance.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Finalizes the dataset when exiting the context manager.

        Parameters:
            exc_type: Exception type, if any occurred during the context.
            exc_value: Exception value, if any occurred during the context.
            traceback: Traceback object, if an exception occurred during the context.
        """
        self.close()
