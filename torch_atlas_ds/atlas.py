import zstandard
import pickle
import json
import numpy as np
from enum import IntEnum
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, NamedTuple



class CompressionStrategy(IntEnum):
    NO_COMPRESSION = 0
    STANDARD_COMPRESSION = 1
    SHARED_DICTIONARY_COMPRESSION = 2
    DICTIONARY_COMPRESSION = 3



class AtlasDatasetShardMetadata(NamedTuple):
    version: int
    block_size: int
    stored_examples: int
    compression_strategy: CompressionStrategy
    compression_level: int
    compression_dict_size: float



class AtlasDatasetShard(Dataset):
    VERSION = 1

    def __init__(self,
                 root: Path | str,
                 mmap_index: bool = True,
                 compression_dict: zstandard.ZstdCompressionDict | None = None
                 ) -> None:
        super().__init__()
        self.root = Path(root)
        self.mmap_index = mmap_index

        self.metadata = AtlasDatasetShardMetadata(**json.loads((self.root / 'meta.json').read_text()))

        if self.metadata.version != AtlasDatasetShard.VERSION:
            raise Exception('The Atlas Dataset Version used in this shard is not supported')

        self.block_index = np.load(self.root / 'index.npy', mmap_mode='r' if mmap_index else None)
        self.data_file = open(self.root / 'data.bin', 'rb')

        if self.metadata.compression_strategy == CompressionStrategy.DICTIONARY_COMPRESSION:
            comp_dict = zstandard.ZstdCompressionDict((self.root / 'zstd_dict.bin').read_bytes())
            self.decompressor = zstandard.ZstdDecompressor(dict_data=comp_dict)
        elif self.metadata.compression_strategy == CompressionStrategy.SHARED_DICTIONARY_COMPRESSION:
            if compression_dict is None:
                raise Exception('When using SHARED_DICTIONARY_COMPRESSION strategy you must provide a compression_dict')
            self.decompressor = zstandard.ZstdDecompressor(dict_data=compression_dict)
        elif self.metadata.compression_strategy == CompressionStrategy.STANDARD_COMPRESSION:
            self.decompressor = zstandard.ZstdDecompressor()
        else:
            self.decompressor = None
    
        self.last_block = None
        self.last_block_idx = None

    def __del__(self) -> None:
        self.data_file.close()
    
    def __len__(self) -> int:
        return self.metadata.stored_examples
    
    def __getitem__(self, index) -> Any:
        if index < 0:
            index = len(self) + index
        
        if index >= len(self):
            raise Exception('Index out of range')
    
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
