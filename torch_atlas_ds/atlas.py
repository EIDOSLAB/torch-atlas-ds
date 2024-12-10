from torch.utils.data import Dataset
from typing import Any, NamedTuple



class AtlasDatasetShardMetadata(NamedTuple):
    version: int
    block_size: int
    stored_examples: int
    compression_enabled: bool
    compression_level: int
    use_compression_dict: bool



class AtlasDatasetShard(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def __len__(self) -> int:
        ...
    
    def __getitem__(self, index) -> Any:
        ...

