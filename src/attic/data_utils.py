# ---- Helper Functions ----
import numpy as np

# ---- One-hot encoding dictionary ----
BASE_DICT = {'A': [1, 0, 0, 0],
             'C': [0, 1, 0, 0],
             'G': [0, 0, 1, 0],
             'T': [0, 0, 0, 1],
             'N': [0, 0, 0, 0]}

def one_hot_encode(seq):
    return np.array([BASE_DICT.get(base.upper(), [0, 0, 0, 0]) for base in seq]).T

def decode_one_hot_old(array):
    index_to_base = {tuple([1, 0, 0, 0]): 'A',
                     tuple([0, 1, 0, 0]): 'C',
                     tuple([0, 0, 1, 0]): 'G',
                     tuple([0, 0, 0, 1]): 'T',
                     tuple([0, 0, 0, 0]): 'N'}  # unknown or padding

    return ''.join(index_to_base.get(tuple(vec), 'N') for vec in array.T)


def decode_one_hot(tensor):
    index_to_base = {
        (1, 0, 0, 0): 'A',
        (0, 1, 0, 0): 'C',
        (0, 0, 1, 0): 'G',
        (0, 0, 0, 1): 'T',
        (0, 0, 0, 0): 'N'  # unknown or padding
    }
    
    # Transpose to iterate over columns (each column is one base)
    return ''.join(index_to_base.get(tuple(tensor[:, i].int().tolist()), 'N') for i in range(tensor.shape[1]))

def make_blocks_no_Pad(start, end, chrom, genome, PADDING=5000, BLOCK_SIZE = 15000):
    region_start = max(0, start - PADDING)
    region_end = end + PADDING
    sequence = genome[chrom][region_start:region_end].seq.upper()
    
    blocks = []
    for i in range(0, len(sequence) - BLOCK_SIZE + 1, PADDING):
        block_seq = sequence[i:i + BLOCK_SIZE]
        blocks.append((region_start + i, region_start + i + BLOCK_SIZE, block_seq))
    return blocks

def make_blocks_Pad(start, end, chrom, genome, PADDING=5000, BLOCK_SIZE=15000):
    ### pad make blocks, very important for 
    ### if region < block_size
    
    region_start = max(0, start - PADDING)
    region_end = end + PADDING
    sequence = genome[chrom][region_start:region_end].seq.upper()

    blocks = []
    if len(sequence) < BLOCK_SIZE:
        # Pad with Ns to reach BLOCK_SIZE
        pad_len = BLOCK_SIZE - len(sequence)
        padded_seq = sequence + 'N' * pad_len
        blocks.append((region_start, region_start + BLOCK_SIZE, padded_seq))
    else:
        # Standard sliding window
        for i in range(0, len(sequence) - BLOCK_SIZE + 1, PADDING):
            block_seq = sequence[i:i + BLOCK_SIZE]
            blocks.append((region_start + i, region_start + i + BLOCK_SIZE, block_seq))
    return blocks

def make_blocks(start, end, chrom, genome, FIXED_PADDING=5000, BLOCK_SIZE=15000):
    """
    Create blocks of sequence for a region. Uses dynamic padding if region is short;
    otherwise uses fixed padding and sliding window.

    Args:
        start (int): Start coordinate of region.
        end (int): End coordinate of region.
        chrom (str): Chromosome name.
        genome (pyfaidx.Fasta): Loaded genome object.
        BLOCK_SIZE (int): Desired block size.
        FIXED_PADDING (int): Padding to apply for long regions.

    Returns:
        List of (block_start, block_end, sequence) tuples.
    """
    
    region_len = abs(end - start)

    if region_len <= BLOCK_SIZE:
        # Case 1: Short region → apply dynamic padding
        pad_total = max(0, BLOCK_SIZE - region_len)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        region_start = max(0, start - pad_left)
        region_end = end + pad_right

        sequence = genome[chrom][region_start:region_end].seq.upper()
        if len(sequence) < BLOCK_SIZE:
            sequence += 'N' * (BLOCK_SIZE - len(sequence))

        return [(region_start, region_end, sequence)]

    else:
        # Case 2: Long region → fixed padding and sliding window
        region_start = max(0, start - FIXED_PADDING)
        region_end = end + FIXED_PADDING

        full_seq = genome[chrom][region_start:region_end].seq.upper()
        blocks = []

        for i in range(0, len(full_seq) - BLOCK_SIZE + 1, FIXED_PADDING):
            block_seq = full_seq[i:i + BLOCK_SIZE]
            block_start = region_start + i
            block_end = block_start + BLOCK_SIZE
            blocks.append((block_start, block_end, block_seq))

        return blocks

def assign_labels(block_start, block_end, psi_dict, PADDING=5000, BLOCK_SIZE = 15000):
    labels = np.zeros((BLOCK_SIZE, 3))  # 12 for 1 tissue (spliced/unspliced/usage) x 4 tissues
    labels[:, 0] = 1 ## default unspliced site for all
    mid_start = PADDING
    mid_end = BLOCK_SIZE - PADDING

    for pos, psi in psi_dict.items():
        if block_start <= pos < block_end:
            rel_pos = pos - block_start
            if 0 <= rel_pos < BLOCK_SIZE:
                # For this example: tissue is heart → positions 0–2
                labels[rel_pos, 0] = 1 if psi < 0.1 else 0
                labels[rel_pos, 1] = 0 if psi < 0.1 else 1
                labels[rel_pos, 2] = 0 if psi < 0.1 else psi
    #return labels[mid_start:mid_end]  # Return only the middle 5000
    return labels.T  # Return Full


def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))
