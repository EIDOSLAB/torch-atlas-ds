import json
import zstandard
from pathlib import Path
from torch_atlas_ds import CompressionStrategy
from tests.common import SENTENCES


class ShardWriterMock:
    def __init__(self,
                 path,
                 block_size,
                 compression_strategy=CompressionStrategy.DICTIONARY_COMPRESSION,
                 compression_level=3,
                 compression_dict=None,
                 compression_dict_size=0.01
                 ) -> None:
        self.path = Path(path)
        self._block_size = block_size
        self._compression_strategy = compression_strategy
        self._compression_level = compression_level
        self._compression_dict = compression_dict
        self._compression_dict_size = compression_dict_size
    
        self.path.mkdir()

        self.examples = []
        self.closed = False

    def add_example(self, example) -> None:
        self.examples.append(example)

    def close(self) -> None:
        assert self.closed == False
        self.closed = True

        new_comp_strat = self._compression_strategy

        if self._compression_strategy == CompressionStrategy.DICTIONARY_COMPRESSION:
            block_count = len(self) // self._block_size + int(bool(len(self) % self._block_size))
            if block_count < 7:
                new_comp_strat = CompressionStrategy.STANDARD_COMPRESSION

        metadata = {
            "version": 1,
            "block_size": self._block_size,
            "stored_examples": len(self),
            "compression_strategy": new_comp_strat,
            "compression_level": self._compression_level,
            "compression_dict_size": self._compression_dict_size
        }

        (self.path / 'meta.json').write_text(json.dumps(metadata, indent=4))
        
        if new_comp_strat == CompressionStrategy.DICTIONARY_COMPRESSION:
            samples = [s.encode() for s in SENTENCES]
            dictionary = zstandard.train_dictionary(dict_size=1024, samples=samples)
            (self.path / 'zstd_dict.bin').write_bytes(dictionary.as_bytes())

    def __len__(self) -> int:
        return len(self.examples)


class ShardWriterMockFactory:
    def __init__(self) -> None:
        self.mocks = []

    def __call__(self, **kwargs) -> ShardWriterMock:
        mock = ShardWriterMock(**kwargs)
        self.mocks.append(mock)
        return mock
