import pytest
from torch_atlas_ds import AtlasDatasetWriter, AtlasDataset
from tests.common import select_random_sample

@pytest.mark.parametrize(
        argnames='num_examples,comp_strat,comp_lev,comp_dict_size,shard_size,block_size,mmap_index',
        argvalues=[
            (1,    0, 3, 0.01, 64, 8, True), # tests with a single example and different compression strategies
            (1,    1, 3, 0.01, 64, 8, True), # tests with a single example and different compression strategies
            (1,    2, 3, 0.01, 64, 8, True), # tests with a single example and different compression strategies
            (1,    3, 3, 0.01, 64, 8, True), # tests with a single example and different compression strategies

            (1,    0, 3, 0.01, 64, 1, True), # tests single example, block size 1, and different compression strategies
            (1,    1, 3, 0.01, 64, 1, True), # tests single example, block size 1, and different compression strategies
            (1,    2, 3, 0.01, 64, 1, True), # tests single example, block size 1, and different compression strategies
            (1,    3, 3, 0.01, 64, 1, True), # tests single example, block size 1, and different compression strategies

            (1,    0, 3, 0.01,  1, 1, True), # tests single example, shard/block size 1, and different compression strategies
            (1,    1, 3, 0.01,  1, 1, True), # tests single example, shard/block size 1, and different compression strategies
            (1,    2, 3, 0.01,  1, 1, True), # tests single example, shard/block size 1, and different compression strategies
            (1,    3, 3, 0.01,  1, 1, True), # tests single example, shard/block size 1, and different compression strategies

            (1,    0, 3, 0.01,  1, 8, True), # tests single example, shard size (1) < block size (8), and different compression strategies
            (1,    1, 3, 0.01,  1, 8, True), # tests single example, shard size (1) < block size (8), and different compression strategies
            (1,    2, 3, 0.01,  1, 8, True), # tests single example, shard size (1) < block size (8), and different compression strategies
            (1,    3, 3, 0.01,  1, 8, True), # tests single example, shard size (1) < block size (8), and different compression strategies

            (10,   0, 3, 0.01,  1, 8, True), # tests multiple examples, shard size (1) < block size (8), and different compression strategies
            (10,   1, 3, 0.01,  1, 8, True), # tests multiple examples, shard size (1) < block size (8), and different compression strategies
            (10,   2, 3, 0.01,  1, 8, True), # tests multiple examples, shard size (1) < block size (8), and different compression strategies
            (10,   3, 3, 0.01,  1, 8, True), # tests multiple examples, shard size (1) < block size (8), and different compression strategies

            (70,   0, 3, 0.01, 32, 8, True), # tests 1 < block count < 7 in all shards and different compression strategies
            (70,   1, 3, 0.01, 32, 8, True), # tests 1 < block count < 7 in all shards and different compression strategies
            (70,   2, 3, 0.01, 32, 8, True), # tests 1 < block count < 7 in all shards and different compression strategies
            (70,   3, 3, 0.01, 32, 8, True), # tests 1 < block count < 7 in all shards and different compression strategies

            (1024, 0, 3, 0.01, 64, 8, True),  # test general case with different compression strategies
            (1024, 1, 3, 0.01, 64, 8, True),  # test general case with different compression strategies
            (1024, 2, 3, 0.01, 64, 8, True),  # test general case with different compression strategies
            (1024, 3, 3, 0.01, 64, 8, True),  # test general case with different compression strategies

            (1025, 0, 3, 0.01, 64, 8, True),  # test general case with different compression strategies
            (1025, 1, 3, 0.01, 64, 8, True),  # test general case with different compression strategies
            (1025, 2, 3, 0.01, 64, 8, True),  # test general case with different compression strategies
            (1025, 3, 3, 0.01, 64, 8, True),  # test general case with different compression strategies

            (1025, 0, 5, 0.01, 64, 8, True),  # test general case with different compression strategies and comp level
            (1025, 1, 5, 0.01, 64, 8, True),  # test general case with different compression strategies and comp level
            (1025, 2, 5, 0.01, 64, 8, True),  # test general case with different compression strategies and comp level
            (1025, 3, 5, 0.01, 64, 8, True),  # test general case with different compression strategies and comp level

            (1025, 0, 3, 0.0, 64, 8, True),  # test general case with different compression strategies and minimum comp dict size
            (1025, 1, 3, 0.0, 64, 8, True),  # test general case with different compression strategies and minimum comp dict size
            (1025, 2, 3, 0.0, 64, 8, True),  # test general case with different compression strategies and minimum comp dict size
            (1025, 3, 3, 0.0, 64, 8, True),  # test general case with different compression strategies and minimum comp dict size

            (1025, 0, 3, 0.01, 64, 8, False),  # test general case with different compression strategies and disable mmap
            (1025, 1, 3, 0.01, 64, 8, False),  # test general case with different compression strategies and disable mmap
            (1025, 2, 3, 0.01, 64, 8, False),  # test general case with different compression strategies and disable mmap
            (1025, 3, 3, 0.01, 64, 8, False),  # test general case with different compression strategies and disable mmap
        ]
)
def test_write_then_read(tmp_path, num_examples, comp_strat, comp_lev, comp_dict_size, shard_size, block_size, mmap_index):
    ds_path = tmp_path / 'atlas_ds'

    writer = AtlasDatasetWriter(
        path=ds_path,
        shard_size=shard_size,
        block_size=block_size,
        compression_strategy=comp_strat,
        compression_level=comp_lev,
        compression_dict_size=comp_dict_size
    )

    samples = select_random_sample(num_examples)

    for sample in samples:
        writer.add_example(sample)
    
    writer.close()

    reader = AtlasDataset(
        root=ds_path,
        mmap_index=mmap_index
    )

    assert len(reader) == len(samples)

    for i in range(len(reader)):
        assert reader[i] == samples[i]

