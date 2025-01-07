import pytest
import json
from unittest.mock import Mock, patch
from torch_atlas_ds import AtlasDatasetWriter, CompressionStrategy
from tests.mocks import ShardWriterMock, ShardWriterMockFactory
from tests.common import select_random_sample

@pytest.mark.parametrize(
        argnames='comp_strat,exp_first_shard_comp_strat',
        argvalues=[
            (0, 0),
            (1, 1),
            (2, 3),
            (3, 3)
        ]
)
@patch('torch_atlas_ds.writer.AtlasDatasetShardWriter')
def test_writer_instantiation(shard_writer_mock: Mock, tmp_path, comp_strat, exp_first_shard_comp_strat):
    ds_path = tmp_path / 'atlas_ds'

    #shard_writer_magic_mock = MagicMock()
    #shard_writer_mock.return_value = shard_writer_magic_mock

    writer = AtlasDatasetWriter(path=ds_path, shard_size=32 * 16, block_size=32, compression_strategy=comp_strat)

    assert writer.path == ds_path
    assert writer.shard_size == 32 * 16
    assert writer.block_size == 32
    assert writer.compression_strategy == comp_strat
    assert writer.compression_level == 3
    assert writer.compression_dict_size == 0.01
    
    assert writer.path.is_dir()

    shard_writer_mock.assert_called_once_with(
        path=ds_path / '0',
        block_size=32,
        compression_strategy=exp_first_shard_comp_strat,
        compression_level=3,
        compression_dict_size=0.01
    )

@pytest.mark.parametrize(
        argnames='num_examples,comp_strat,exp_comp_strat,shard_size,block_size',
        argvalues=[
            (1,    0, 0, 64, 8), # tests with a single example and different compression strategies
            (1,    1, 1, 64, 8), # tests with a single example and different compression strategies
            (1,    2, 1, 64, 8), # tests with a single example and different compression strategies
            (1,    3, 3, 64, 8), # tests with a single example and different compression strategies

            (1,    0, 0, 64, 1), # tests single example, block size 1, and different compression strategies
            (1,    1, 1, 64, 1), # tests single example, block size 1, and different compression strategies
            (1,    2, 1, 64, 1), # tests single example, block size 1, and different compression strategies
            (1,    3, 3, 64, 1), # tests single example, block size 1, and different compression strategies

            (1,    0, 0,  1, 1), # tests single example, shard/block size 1, and different compression strategies
            (1,    1, 1,  1, 1), # tests single example, shard/block size 1, and different compression strategies
            (1,    2, 1,  1, 1), # tests single example, shard/block size 1, and different compression strategies
            (1,    3, 3,  1, 1), # tests single example, shard/block size 1, and different compression strategies

            (1,    0, 0,  1, 8), # tests single example, shard size (1) < block size (8), and different compression strategies
            (1,    1, 1,  1, 8), # tests single example, shard size (1) < block size (8), and different compression strategies
            (1,    2, 1,  1, 8), # tests single example, shard size (1) < block size (8), and different compression strategies
            (1,    3, 3,  1, 8), # tests single example, shard size (1) < block size (8), and different compression strategies

            (10,   0, 0,  1, 8), # tests multiple examples, shard size (1) < block size (8), and different compression strategies
            (10,   1, 1,  1, 8), # tests multiple examples, shard size (1) < block size (8), and different compression strategies
            (10,   2, 1,  1, 8), # tests multiple examples, shard size (1) < block size (8), and different compression strategies
            (10,   3, 3,  1, 8), # tests multiple examples, shard size (1) < block size (8), and different compression strategies

            (70,   0, 0, 32, 8), # tests 1 < block count < 7 in all shards and different compression strategies
            (70,   1, 1, 32, 8), # tests 1 < block count < 7 in all shards and different compression strategies
            (70,   2, 1, 32, 8), # tests 1 < block count < 7 in all shards and different compression strategies
            (70,   3, 3, 32, 8), # tests 1 < block count < 7 in all shards and different compression strategies

            (1024, 0, 0, 64, 8),  # test general case with different compression strategies
            (1024, 1, 1, 64, 8),  # test general case with different compression strategies
            (1024, 2, 2, 64, 8),  # test general case with different compression strategies
            (1024, 3, 3, 64, 8),  # test general case with different compression strategies

            (1025, 0, 0, 64, 8),  # test general case with different compression strategies
            (1025, 1, 1, 64, 8),  # test general case with different compression strategies
            (1025, 2, 2, 64, 8),  # test general case with different compression strategies
            (1025, 3, 3, 64, 8),  # test general case with different compression strategies
        ]
)
@patch('torch_atlas_ds.writer.AtlasDatasetShardWriter', new_callable=ShardWriterMockFactory)
def test_writer_add_examples_and_close(shard_writer_mock: ShardWriterMockFactory, tmp_path, num_examples, comp_strat, exp_comp_strat, shard_size, block_size):
    ds_path = tmp_path / 'atlas_ds'

    writer = AtlasDatasetWriter(
        path=ds_path,
        shard_size=shard_size,
        block_size=block_size,
        compression_strategy=comp_strat,
    )

    samples = select_random_sample(num_examples)

    for sample in samples:
        writer.add_example(sample)
    
    writer.close()

    expected_shard_sizes = [shard_size] * (num_examples // shard_size)
    
    if num_examples % shard_size != 0:
        expected_shard_sizes.append(num_examples % shard_size)

    assert len(shard_writer_mock.mocks) == len(expected_shard_sizes)

    added_samples = []
    for i, mock in enumerate(shard_writer_mock.mocks):
        assert len(mock) == expected_shard_sizes[i]
        assert mock.closed
        for ex in mock.examples:
            added_samples.append(ex)
    
    assert added_samples == samples

    metadata = json.loads((ds_path / 'meta.json').read_text())

    assert metadata == {
        'version': AtlasDatasetWriter.VERSION,
        'shard_sizes': expected_shard_sizes,
        'compression_strategy': exp_comp_strat
    }

    exp_folder_names = []
    len_folder_names = len(str(len(expected_shard_sizes) - 1))

    for i in range(len(expected_shard_sizes)):
        exp_folder_names.append(f'{i:0>{len_folder_names}}')

    actual_folder_names = []
    for p in ds_path.iterdir():
        if p.is_dir():
            actual_folder_names.append(p.name)
    actual_folder_names = sorted(actual_folder_names)

    assert actual_folder_names == exp_folder_names

    for i, shard_name in enumerate(exp_folder_names):
        shard_meta = json.loads((ds_path / shard_name / 'meta.json').read_text())
        exp_shard_comp_strat = exp_comp_strat

        shard_block_count = expected_shard_sizes[i] // block_size + int(bool(expected_shard_sizes[i] % block_size))

        if comp_strat == CompressionStrategy.DICTIONARY_COMPRESSION:
            if shard_block_count < 7:
                exp_shard_comp_strat = 1
        
        assert shard_meta['compression_strategy'] == exp_shard_comp_strat

    if exp_comp_strat == CompressionStrategy.SHARED_DICTIONARY_COMPRESSION:
        shard_dict_path = ds_path / exp_folder_names[0] / 'zstd_dict.bin'
        shared_dict_path = ds_path / 'zstd_dict.bin'
        assert not shard_dict_path.is_file()
        assert shared_dict_path.is_file()
