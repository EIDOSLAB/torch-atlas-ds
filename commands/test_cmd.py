import torch
from pathlib import Path
from torch_atlas_ds.writer import AtlasDatasetShardWriter

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
                           compress: bool = True,
                           compression_level: int = 3,
                           use_compression_dict = True,
                           compression_dict_size: float = 0.01
                           ) -> None:
        

        with AtlasDatasetShardWriter(output_path, block_size, compress, compression_level, use_compression_dict, compression_dict_size) as shard_writer:
            with open(input_text_file, 'r') as text_file:
                for line in text_file:
                    shard_writer.add_example(line.rstrip())
        
        print('done')
