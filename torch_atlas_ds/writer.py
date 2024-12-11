import zstandard
import pickle
import json
import bisect
import numpy as np
from pathlib import Path
from typing import Any
from torch_atlas_ds.atlas import AtlasDatasetShardMetadata, CompressionStrategy





class AtlasDatasetShardWriter:
    VERSION = 1

    def __init__(self,
                 path: Path | str,
                 block_size: int,
                 compression_strategy: CompressionStrategy = CompressionStrategy.DICTIONARY_COMPRESSION,
                 compression_level: int = 3,
                 compression_dict: zstandard.ZstdCompressionDict | None = None,
                 compression_dict_size: float = 0.01
                 ) -> None:
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
                
                for i in range(len(self.pickled_blocks)):
                    block = self.pickled_blocks[i]
                    compressed = cctx.compress(block)
                    self.pickled_blocks[i] = compressed

                (self.path / 'zstd_dict.bin').write_bytes(compression_dict.as_bytes())
                
                del cctx
                del compression_dict
            else:
                self._update_metadata('compression_strategy', CompressionStrategy.STANDARD_COMPRESSION)
                self.compressor = zstandard.ZstdCompressor(level=self.metadata.compression_level)
            
            for block in self.pickled_blocks:
                self._write_block(block)
    
        if self.data_file is not None:
            self.data_file.close()
        
        # add the size of the whole shard in bytes as last element of the index
        # makes it easier to calculate the size of the last block
        self.index.append(self.shard_size_bytes)
        
        self._save_index()
        
        (self.path / 'meta.json').write_text(json.dumps(self.metadata._asdict(), indent=4))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
