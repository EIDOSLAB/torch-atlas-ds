import torch
import zstandard
from pathlib import Path
from torch_atlas_ds.writer import AtlasDatasetShardWriter, CompressionStrategy, AtlasDatasetWriter

class TestCommands:
    """
    Commands used to run tests.
    """

    def gpu(self) -> None:
        """
        Checks if GPUs are available and working correctly.
        """
        cuda_av = torch.cuda.is_available()

        print(f'CUDA Available: {cuda_av}')

        if cuda_av:
            for i in range(torch.cuda.device_count()):
                print(f'Found GPU device: {torch.cuda.get_device_name(i)}')

                a = torch.rand(10, 10).to(f'cuda:{i}')
                b = torch.rand(10, 20).to(f'cuda:{i}')
                r = a @ b
            
            print('GPU Test successful')


    def write_single_shard(self,
                           input_text_file: Path | str,
                           output_path: Path | str,
                           block_size: int,
                           compression: str = 'dictionary',
                           compression_level: int = 3,
                           compression_dict_size: float = 0.01,
                           compression_dict_path: Path | str | None = None
                           ) -> None:
        
        compression_strategy = CompressionStrategy[compression.upper() + '_COMPRESSION']

        params = {
            "path": Path(output_path),
            "block_size": block_size,
            "compression_strategy": compression_strategy,
            "compression_level": compression_level,
            "compression_dict_size": compression_dict_size
        }

        if compression_strategy == CompressionStrategy.SHARED_DICTIONARY_COMPRESSION:
            assert compression_dict_path is not None, 'compression_dict_path was not provided'
            params['compression_dict'] = zstandard.ZstdCompressionDict(Path(compression_dict_path).read_bytes())

        with AtlasDatasetShardWriter(**params) as shard_writer:
            with open(input_text_file, 'r') as text_file:
                for line in text_file:
                    shard_writer.add_example(line.rstrip())
        
        print('done')


    def write_atlas_dataset(self,
                            input_text_file: Path | str,
                            output_path: Path | str,
                            shard_size: int,
                            block_size: int,
                            compression: str = 'dictionary',
                            compression_level: int = 3,
                            compression_dict_size: float = 0.01
                            ) -> None:

        compression_strategy = CompressionStrategy[compression.upper() + '_COMPRESSION']

        params = {
            "path": Path(output_path),
            "shard_size": shard_size,
            "block_size": block_size,
            "compression_strategy": compression_strategy,
            "compression_level": compression_level,
            "compression_dict_size": compression_dict_size
        }

        with AtlasDatasetWriter(**params) as atlas_writer:
            with open(input_text_file, 'r') as text_file:
                for line in text_file:
                    atlas_writer.add_example(line.rstrip())

        print('done')
