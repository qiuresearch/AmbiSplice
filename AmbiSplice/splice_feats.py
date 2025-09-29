# ---- Helper Functions ----
import numpy as np

# ---- One-hot encoding dictionary ----
BASE_DICT = {'A': [1, 0, 0, 0],
             'C': [0, 1, 0, 0],
             'G': [0, 0, 1, 0],
             'T': [0, 0, 0, 1],
             'N': [0, 0, 0, 0]}

def one_hot_encode_CL(seq): # CL --> return in shape [C, L]
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


def sprinkle_sites_onto_vectors(rna_sites, debug=False):
    """ Convert splice site positions and labels into vectors aligned with the RNA sequence """

    rna_seq = rna_sites['seq']
    # classification labels: 0: no site, 1: donor, 2: acceptor, 3: hybrid
    vec_cls = np.zeros(len(rna_seq), dtype=int)
    vec_cls_odds = np.zeros(len(rna_seq), dtype=float)
    vec_cls_mask = np.zeros(len(rna_seq), dtype=int)
    # pos is 0-based, pretty slow way to do this
    for pos, site_type, site_odds in zip(rna_sites['cls_pos'], rna_sites['cls_type'], rna_sites['cls_odds']):
        vec_cls_odds[pos] = site_odds
        vec_cls_mask[pos] = 1
        if site_type == 'donor':
            vec_cls[pos] = 1 # donor
        elif site_type == 'acceptor':
            vec_cls[pos] = 2 # acceptor
        elif site_type == 'hybrid':
            vec_cls[pos] = 3 # hybrid
        else:
            raise ValueError(f"Invalid site type: {site_type}")

    # psi values (initialize with zeros!)
    vec_psi = np.zeros(len(rna_seq), dtype=float)
    vec_psi_std = np.zeros(len(rna_seq), dtype=float)
    vec_psi_mask = np.zeros(len(rna_seq), dtype=int)
    # pos is 0-based, pretty slow way to do this
    for pos, psi, psi_std in zip(rna_sites['psi_pos'], rna_sites['psi'], rna_sites['psi_std']):
        vec_psi[pos] = psi
        vec_psi_std[pos] = psi_std
        vec_psi_mask[pos] = 1

    return {'seq': rna_seq,
            'cls': vec_cls,
            'cls_odds': vec_cls_odds,
            'cls_mask': vec_cls_mask,
            'psi': vec_psi,
            'psi_std': vec_psi_std,
            'psi_mask': vec_psi_mask,
            }


def get_train_feats_single_rna(rna_feats,
                               chunk_size=5000, flank_size=5000,
                               min_sites=0, min_usage=0,
                               debug=False):
    """ Generate training features for a single RNA transcript by chunking the sequence
    Args:
        rna_feats: dict, output of sprinkle_sites_onto_vectors
        rna_seq: str, RNA sequence"""
    rna_seq = rna_feats['seq']
    rna_cls = rna_feats['cls']
    rna_cls_odds = rna_feats['cls_odds']
    rna_cls_mask = rna_feats['cls_mask']
    rna_psi = rna_feats['psi']
    rna_psi_std = rna_feats['psi_std']
    rna_psi_mask = rna_feats['psi_mask']
    
    total_size = chunk_size + 2 * flank_size

    # divide gene_lsv into chunks of chunk_size 
    seq_len = len(rna_seq)
    chunk_starts = np.arange(0, seq_len, chunk_size)
    chunk_starts[-1] = max([0, seq_len - chunk_size])  # ensure last chunk reaches the end

    chunk_ends = chunk_starts + chunk_size
    chunk_ends[-1] = seq_len  # ensure last chunk reaches the end

    if debug:
        print(f"Seq Length: {seq_len}\nChunk starts: {chunk_starts}\nChunk ends: {chunk_ends}")

    train_feats = []
    for chunk_start, chunk_end in zip(chunk_starts, chunk_ends):
        # chunk_end - chunk_start <= chunk_size always (< if gene is shorter than chunk_size)
        train_feat = {'chunk_start': chunk_start, 'chunk_end': chunk_end}

        # seq_start and seq_end include flanking regions
        seq_start = max(0, chunk_start - flank_size)
        seq_end = min(seq_len, chunk_end + flank_size)

        pad_size = total_size - (seq_end - seq_start)
        left_pad = flank_size - (chunk_start - seq_start)
        right_pad = pad_size - left_pad
        
        # pad with 'N's if needed for input X
        train_feat['seq'] = 'N' * left_pad + rna_seq[seq_start:seq_end] + 'N' * right_pad
        train_feat['seq_onehot'] = one_hot_encode_CL(train_feat['seq']).astype(np.float32)  # shape [4, total_size]

        train_feat['cls'] = np.pad(rna_cls[seq_start:seq_end], (left_pad, right_pad), 'constant')
        train_feat['cls_odds'] = np.pad(rna_cls_odds[seq_start:seq_end], (left_pad, right_pad), 'constant')
        train_feat['cls_mask'] = np.pad(rna_cls_mask[seq_start:seq_end], (left_pad, right_pad), 'constant')
        
        train_feat['psi'] = np.pad(rna_psi[seq_start:seq_end], (left_pad, right_pad), 'constant')
        train_feat['psi_std'] = np.pad(rna_psi_std[seq_start:seq_end], (left_pad, right_pad), 'constant')
        train_feat['psi_mask'] = np.pad(rna_psi_mask[seq_start:seq_end], (left_pad, right_pad), 'constant')

        # target Y only has the middle chunk_size region excluding flanks
        train_feat['cls'] = train_feat['cls'][flank_size:flank_size + chunk_size]
        train_feat['cls_odds'] = train_feat['cls_odds'][flank_size:flank_size + chunk_size]
        train_feat['cls_mask'] = train_feat['cls_mask'][flank_size:flank_size + chunk_size] # not used currently
        train_feat['psi'] = train_feat['psi'][flank_size:flank_size + chunk_size]
        train_feat['psi_std'] = train_feat['psi_std'][flank_size:flank_size + chunk_size]
        train_feat['psi_mask'] = train_feat['psi_mask'][flank_size:flank_size + chunk_size] # not used currently

        # parts of Y labels may be padding if the entire sequence is shorter than chunk_size
        # train_feat['chunk_mask'] = np.zeros(chunk_size, dtype=np.int32)
        # the actual start of the seq[start:end] within the chunk_size region of seq
        # y_start = left_pad + (chunk_start - seq_start) - flank_size
        # y_end = y_start + (chunk_end - chunk_start)
        # train_feat['chunk_mask'][y_start:y_end] = 1

        if min_sites and np.sum(train_feat['cls'][:chunk_end-chunk_start]) <= min_sites:
            # print(f"Skipping gene_id: {train_feat['gene_id']}; start: {train_feat['start']}; end: {train_feat['end']} with <= {min_sites} splice sites!")
            continue

        if min_usage and np.sum(train_feat['psi'][:chunk_end-chunk_start]) < min_usage:
            # print(f"Skipping gene_id: {train_feat['gene_id']}; start: {train_feat['start']}; end: {train_feat['end']} with <= {min_usage} usage!")
            continue

        train_feats.append(train_feat)

    return train_feats


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
