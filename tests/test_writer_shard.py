import pytest
import zstandard
import json
import numpy as np
import pickle
import random
import math
from torch_atlas_ds import AtlasDatasetShardWriter, CompressionStrategy

SENTENCES = [
    "The sun rises in the east and sets in the west.",
    "Cats are known for their graceful movements.",
    "She enjoys reading books in her free time.",
    "A quick brown fox jumps over the lazy dog.",
    "The park was filled with children playing games.",
    "He prefers tea over coffee in the morning.",
    "The train arrived at the station right on time.",
    "Winter is the coldest season of the year."
]


@pytest.mark.parametrize(
        argnames='comp_strat,expect_compressor',
        argvalues=[
            (0, False),
            (1, True),
            (2, True),
            (3, False)
        ]
)
def test_writer_instantiation(tmp_path, comp_strat, expect_compressor):
    shard_path = tmp_path / 'shard'

    dictionary = None
    if comp_strat == CompressionStrategy.SHARED_DICTIONARY_COMPRESSION:
        samples = [s.encode() for s in SENTENCES]
        dictionary = zstandard.train_dictionary(dict_size=1024, samples=samples)

    writer = AtlasDatasetShardWriter(
        path=shard_path,
        block_size=32,
        compression_strategy=comp_strat,
        compression_dict=dictionary
    )

    assert writer.path == shard_path
    assert writer.metadata.version == 1
    assert writer.metadata.block_size == 32
    assert writer.metadata.compression_strategy == comp_strat
    assert writer.metadata.compression_level == 3
    assert writer.metadata.compression_dict_size == 0.01
    assert shard_path.is_dir()
    assert writer.stored_examples == 0
    assert (writer.compressor is not None) == expect_compressor


@pytest.mark.parametrize(
        argnames='block_size,comp_strat,comp_dict_size,expected',
        argvalues=[
            (32, 2, 0.1,  Exception),
            (0,  3, 0.1,  AssertionError),
            (-1, 3, 0.1,  AssertionError),
            (32, 3, -0.1, AssertionError),
            (32, 3, 1.1,  AssertionError)
        ]
)
def test_writer_instantiation_exceptions(tmp_path, block_size, comp_strat, comp_dict_size, expected):
    shard_path = tmp_path / 'shard'

    with pytest.raises(expected):
        writer = AtlasDatasetShardWriter(
            path=shard_path,
            block_size=block_size,
            compression_strategy=comp_strat,
            compression_dict_size=comp_dict_size
        )


@pytest.mark.parametrize(
        argnames='num_examples,comp_strat,exp_comp_strat,block_size,comp_lev,comp_dict_size',
        argvalues=[
            (1,   0, 0, 32, 3, 0.1), # tests with a single example and different compression strategies
            (1,   1, 1, 32, 3, 0.1), # tests with a single example and different compression strategies
            (1,   2, 2, 32, 3, 0.1), # tests with a single example and different compression strategies
            (1,   3, 1, 32, 3, 0.1), # tests with a single example and different compression strategies

            (1,   0, 0, 1, 3, 0.1 ), # tests with a single example and block_size == 1 and different compression strategies
            (1,   1, 1, 1, 3, 0.1 ), # tests with a single example and block_size == 1 and different compression strategies
            (1,   2, 2, 1, 3, 0.1 ), # tests with a single example and block_size == 1 and different compression strategies
            (1,   3, 1, 1, 3, 0.1 ), # tests with a single example and block_size == 1 and different compression strategies

            (3,   0, 0, 1, 3, 0.1 ), # tests with some examples and block_size == 1 and different compression strategies
            (3,   1, 1, 1, 3, 0.1 ), # tests with some examples and block_size == 1 and different compression strategies
            (3,   2, 2, 1, 3, 0.1 ), # tests with some examples and block_size == 1 and different compression strategies
            (3,   3, 1, 1, 3, 0.1 ), # tests with some examples and block_size == 1 and different compression strategies

            (192, 3, 1, 32, 3, 0.1), # with 6 blocks it should fall back to STANDARD_COMPRESSION

            (224, 3, 3, 32, 3, 0.1), # with 7 blocks it should not fall back to STANDARD_COMPRESSION
            (224, 0, 0, 32, 3, 0.1), # tests with full blocks and different compression strategies
            (224, 1, 1, 32, 3, 0.1), # tests with full blocks and different compression strategies
            (224, 2, 2, 32, 3, 0.1), # tests with full blocks and different compression strategies
            
            (225, 0, 0, 32, 3, 0.1), # tests where the last block is not full (with different compression strategies)
            (225, 1, 1, 32, 3, 0.1), # tests where the last block is not full (with different compression strategies)
            (225, 2, 2, 32, 3, 0.1), # tests where the last block is not full (with different compression strategies)
            (225, 3, 3, 32, 3, 0.1), # tests where the last block is not full (with different compression strategies)

            (225, 1, 1, 32, 6, 0.1), # test a different compression level
            (225, 2, 2, 32, 6, 0.1), # test a different compression level
            (225, 3, 3, 32, 6, 0.1), # test a different compression level

            (225, 3, 3, 32, 3, 0.0), # test minimum sized dictionary (1024 bytes)

            (100000, 0, 0, 32, 3, 0.1), # tests big shards (with different compression strategies)
            (100000, 1, 1, 32, 3, 0.1), # tests big shards (with different compression strategies)
            (100000, 2, 2, 32, 3, 0.1), # tests big shards (with different compression strategies)
            (100000, 3, 3, 32, 3, 0.1), # tests big shards (with different compression strategies)
        ]
)
def test_writer_add_examples_and_close(tmp_path, num_examples, comp_strat, exp_comp_strat, block_size, comp_lev, comp_dict_size):
    shard_path = tmp_path / 'shard'

    dictionary = None
    if comp_strat == CompressionStrategy.SHARED_DICTIONARY_COMPRESSION:
        samples = [s.encode() for s in SENTENCES]
        dictionary = zstandard.train_dictionary(dict_size=1024, samples=samples)
    
    writer = AtlasDatasetShardWriter(
        path=shard_path,
        block_size=block_size,
        compression_strategy=comp_strat,
        compression_level=comp_lev,
        compression_dict=dictionary,
        compression_dict_size=comp_dict_size
    )

    random.seed(117)
    samples_multiplier = math.ceil(num_examples / len(SENTENCES))
    samples = SENTENCES * samples_multiplier
    random.shuffle(samples)
    samples = samples[:num_examples]
    assert len(samples) == num_examples

    for i, sample in enumerate(samples):
        assert writer.stored_examples == i
        writer.add_example(sample)
    
    assert writer.stored_examples == num_examples

    writer.close()

    assert writer.stored_examples == num_examples

    metadata_path = shard_path / 'meta.json'
    assert metadata_path.is_file()

    metadata = json.loads(metadata_path.read_text())

    assert metadata == {
        "version": 1,
        "block_size": block_size,
        "stored_examples": num_examples,
        "compression_strategy": exp_comp_strat,
        "compression_level": comp_lev,
        "compression_dict_size": comp_dict_size
    }

    index_path = shard_path / 'index.npy'
    assert index_path.is_file()

    data_path = shard_path / 'data.bin'
    assert data_path.is_file()

    index = np.load(index_path)
    data = data_path.read_bytes()

    expected_block_count = num_examples // block_size + int(bool(num_examples % block_size))

    assert len(index) == expected_block_count + 1
    assert index[0] == 0
    assert index[-1] == len(data)

    dict_path = shard_path / 'zstd_dict.bin'
    assert (exp_comp_strat == CompressionStrategy.DICTIONARY_COMPRESSION) != (not dict_path.is_file())

    if exp_comp_strat == CompressionStrategy.SHARED_DICTIONARY_COMPRESSION:
        cctx = zstandard.ZstdDecompressor(dict_data=dictionary)
    elif exp_comp_strat == CompressionStrategy.DICTIONARY_COMPRESSION:
        cctx = zstandard.ZstdDecompressor(dict_data=zstandard.ZstdCompressionDict(dict_path.read_bytes()))
    elif exp_comp_strat == CompressionStrategy.STANDARD_COMPRESSION:
        cctx = zstandard.ZstdDecompressor()
    else:
        class MockZstandard:
            def decompress(self, x):
                return x
        cctx = MockZstandard()
    
    last_block_size = num_examples % block_size
    if last_block_size == 0:
        last_block_size = block_size

    for i in range(expected_block_count):
        offset = index[i]
        next_offset = index[i+1]
        block_data = data[offset:next_offset]
        decompressed = cctx.decompress(block_data)
        extracted_examples = pickle.loads(decompressed)

        expected_block_size = block_size
        if i == (expected_block_count - 1):
            expected_block_size = last_block_size

        assert isinstance(extracted_examples, list)
        assert len(extracted_examples) == expected_block_size

        for j in range(len(extracted_examples)):
            sample_idx = i * block_size + j
            assert extracted_examples[j] == samples[sample_idx]
