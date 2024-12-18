# Atlas Dataset Format (Version 1)

The **Atlas Dataset Format** is designed for fast random access and efficient storage of large-scale datasets. It divides the dataset into **shards** and each shard into **blocks**. It supports different compression strategies, including the option of using no compression. This document describes the structure of an Atlas Dataset.

---

## Directory Structure

An Atlas dataset is stored in a directory with the following structure:

```
dataset_root/
├── meta.json                # Metadata for the entire dataset
├── zstd_dict.bin            # (Optional) Shared compression dictionary (if present, it's used for all the shards)
├── 00/                      # shard 0 (a shard is a folder)
│   ├── meta.json            # Metadata for shard 0
│   ├── data.bin             # Binary data for shard 0 (stores the examples)
│   ├── index.npy            # Index for locating blocks within data.bin
│   └── zstd_dict.bin        # (Optional) Compression dictionary for shard 0
├── 01/                      # shard 1
│   ├── meta.json
│   ├── data.bin
│   ├── index.npy
│   └── zstd_dict.bin
...
```

The names for the shard folders have always the same length and are always numbers that increment by 1 for each shard that is added to the dataset (starting from 0). After the dataset is created, the shard folder names are padded with 0s to the left in order to be the same length.

## Components

### **1. Dataset Metadata (`meta.json`)**
Located in the dataset root directory, this file contains metadata describing the entire dataset.

#### Example (`meta.json`):
```json
{
    "version": 1,
    "shard_sizes": [1000, 1000, 800],
    "compression_strategy": 2
}
```

- **Fields:**
  - `version`: The format version of the dataset (integer).
  - `shard_sizes`: A list of integers, where each value represents the number of examples in a corresponding shard.
  - `compression_strategy`: Enum indicating the compression strategy:
    - `0`: No compression.
    - `1`: Standard compression.
    - `2`: Shared dictionary compression.
    - `3`: Individual dictionary compression.

Currently `shard_sizes` is not used while reading the dataset, the shard sizes are directly read from the meta.json file of each shard. This field can be useful in the future to implement other features for cases where some of the shards are not available but we still want to use the same indices for the examples that we would use if all the shards were available.

---

### **2. Shards**
Each shard is a directory containing the data for a subset of the dataset. It includes three key files:

#### **a) Shard Metadata (`meta.json`)**
Each shard has its own `meta.json` file that describes how the shard is stored.

#### Example (`shard_id/meta.json`):
```json
{
    "version": 1,
    "block_size": 100,
    "stored_examples": 1200,
    "compression_strategy": 3,
    "compression_level": 3,
    "compression_dict_size": 0.01
}
```

- **Fields:**
  - `version`: The format version (integer).
  - `block_size`: Number of examples per block (integer).
  - `stored_examples`: Total examples in the shard (integer).
  - `compression_strategy`: Compression strategy for this shard (see above).
  - `compression_level`: Compression level used for Zstandard.
  - `compression_dict_size`: Size of the compression dictionary as a fraction of the uncompressed data (float).

`compression_level` and `compression_dict_size` are reported in the metadata just for reference, they are not actually used while reading the dataset.

---

#### **b) Data File (`shard_id/data.bin`)**
A binary file that stores the serialized examples for this shard. This file is a sequence of blocks one after the other, each block contains up to `block_size` examples and is created by doing the following:

1) The examples that should be put in the block are stored in a python list.
2) The list is serialized with pickle.
3) If compression is desired, the serialized list is compressed according to the specified compression level and strategy using the Zstandard algorithm.
4) The sequence of bytes we obtain from the above steps is called block, and we append it to `data.bin`.

Notice that all blocks except the last one will have a number of examples equal to the `block_size`, the last block will store the remaining examples.

---

#### **c) Index File (`shard_id/index.npy`)**
A NumPy array storing offsets for each block in the `data.bin` file. It allows for efficient random access to any block.

- **Contents:**
  - If there are `N` blocks, the array contains `N+1` entries:
    - The `i-th` entry gives the byte offset of the `i-th` block in `data.bin`.
    - The final entry gives the total size (in bytes) of the `data.bin` file.

The dtype of this array is an unsigned integer of size large enough to store the last integer in the array, the size is chosen between 8, 16, 32, or 64 bits.

---

#### **d) Compression Dictionary (`shard_id/zstd_dict.bin`)** *(Optional)*
- Present if the shard uses dictionary-based compression.
- A custom compression dictionary trained on the shard's data to improve compression ratios.

---

## Compression Strategies

### **CompressionStrategy Enum**
Defines how data is compressed within the dataset:

| Value | Name                          | Description                                                                       |
|-------|-------------------------------|-----------------------------------------------------------------------------------|
| `0`   | NO_COMPRESSION                | No compression applied.                                                           |
| `1`   | STANDARD_COMPRESSION          | Zstandard compression applied to each block (no dictionary is used).              |
| `2`   | SHARED_DICTIONARY_COMPRESSION | All shards share a common compression dictionary (`zstd_dict.bin`).               |
| `3`   | DICTIONARY_COMPRESSION        | Each shard uses its own custom compression dictionary (`shard_id/zstd_dict.bin`). |

---

## Example of Data Access

1. **Locating the i-th example:**
   
   If index `i` is requested, we calculate the following indices:
   - `shard_idx`: index of the shard that contains the `i`-th example.
   - `block_idx`: index of the block within the selected shard that contains the `i`-th example.
   - `example_idx`: index of the example within the selected block.

2. **Reading the Index:**
   - If not done already, load or memory map the `index.npy` file of the shard number `shard_idx`
   - Read the `block_idx`-th offset (`block_offset`) stored in `index.npy`.
   - Read also the next offset (`block_idx+1`).
   - Use both offsets to calculate the size in bytes of the block (`block_len`).

3. **Retrieving a Block:**
   - Use `block_offset` to seek to the desired block in the shard's `data.bin` file.
   - Read `block_len` bytes from the shard's `data.bin` file.
   - If compressed, decompress the block using the appropriate Zstandard decompressor.

4. **Parsing the Block:**
   - Each block is serialized using Python's `pickle`. Deserialize it to retrieve the list of examples stored in the block.

5. **Returning the example:**
   - Return the example at index `example_idx` in the previously obtained list of examples.

For each shard the last block that was read and parsed can be cached to speed up sequential reads.

---

## FAQ

### **Why use blocks instead of storing each example separately?**
Blocks reduce the overhead of compression and reduce the size of the index. As explained above, caching a block can also improve sequential reads.

### **What happens if I train a dictionary on a small dataset?**
Dictionary training in zstandard requires at least 7 examples. This means that if the shard has less than 7 blocks, the dictionary cannot be trained, and the format will fall back to `STANDARD_COMPRESSION` strategy.

### **What are appropriate values for shard_size and block_size?**

If using dictionary compression, you have to choose a `shard_size` and a `block_size` such that there are at least 7 blocks in the shard, otherwise as explained above, dictionary compression will not be used.

Having said this, if your dataset is for example 100 GB in size, a good shard size may be the one that splits the dataset into 100 shards of 1 GB. If the dataset is 1 TB, you may choose to have 100 shards of 10 GB, or if you prefer smaller shards you can have 1000 shards of 1 GB.

The `block_size` is important for determining the balance between the compression ratio and the speed, bigger block_sizes compress better especially when not using a compression dictionary, but bigger block sizes also slow down random access to the examples, because you have to read the same block `block_size` times during an epoch. If reading sequentially big block sizes may be better since the last read block is cached. A suggestion is to try with powers of 2 between 8 and 1024 with a subset of your dataset and picking the one that achieves better speed.

### **What compression strategy should I use? What compression level and what compression_dict_size?**

Regarding compression strategies, use:
- `NO_COMPRESSION`: when the data you are storing is already compressed or difficult to compress.
- `STANDARD_COMPRESSION`: when you want to create the dataset faster without training the dictionary, and when compression ratios are not important, or when you have big blocks that would benefit less from dictionary compression.
- `SHARED_DICTIONARY_COMPRESSION`: in most cases, this is the recommended strategy because you have to train only 1 dictionary on the data contained in the first shard, and then the dictionary is used by all other shards. This usually achieves optimal compression ratios.
- `DICTIONARY_COMPRESSION`: this is the slowest strategy when writing the datasets, use it only if you need very high compression ratios and when the distribution of the examples in the first shard does not match the distribution in the other shards. Usually, if the distributions match, `SHARED_DICTIONARY_COMPRESSION` achieves the same compression ratio of `DICTIONARY_COMPRESSION` but with less overhead (store only 1 dictionary instead of many).

Regarding the compression level, by default zstandard uses 3, which strikes a good balance between speed and compression ratio. Keep in mind that usually zstandard decompression speed is not affected a lot by the compression level, while the compression speed is very much affected. If you care more about read speeds and compression ratio than write speeds, use a higher compression level (the maximum is 22).

The `compression_dict_size` is set by default at 0.01, usually this is overkill especially for big shards, the suggestion is to leave this alone for smaller shards (1-3 GB or less) and to decrease it for bigger shards. Decrease it more heavily when using `DICTIONARY_COMPRESSION` since each shard will have it's own dictionary. For `SHARED_DICTIONARY_COMPRESSION` you can leave this alone, or increase it if you find that your dataset is so big that needs a bigger dictionary.