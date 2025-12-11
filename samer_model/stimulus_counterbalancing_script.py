import numpy as np
def find_counterbalanced_quads(lexicon, min_sem_overlap=1):
    """
    Find sets of 4 words [a, b, c, d] where:
    - a & b are sem-related (not orth-related)
    - c & d are sem-related (not orth-related)
    - {a,b} and {c,d} share NO sem or orth features
    
    Soft constraint: prefer higher semantic overlap within pairs.
    """
    
    # shared_feats = lexicon.semfeatmatrix.T @ lexicon.semfeatmatrix
    shared_feats = np.load('./helper_txt_files/shared_feats.npy')
    diag_mask = np.ones(shared_feats.shape) - np.eye(shared_feats.shape[0])
    only_shared_features = shared_feats * diag_mask
    
    orthrelated_matrix = lexicon.orthmatrix.T @ lexicon.orthmatrix
    
    n_words = lexicon.size
    
    # Step 1: Find ALL valid sem-related pairs with their overlap strength
    sem_pairs = []
    
    for i in range(n_words):
        for j in range(i+1, n_words):
            sem_overlap = only_shared_features[i, j]
            orth_overlap = orthrelated_matrix[i, j]
            
            if sem_overlap >= min_sem_overlap and orth_overlap == 0:
                sem_pairs.append((i, j, sem_overlap))
    
    # Sort by semantic overlap (strongest first)
    sem_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Found {len(sem_pairs)} valid sem-related pairs")
    print(f"Overlap distribution: "
          f"max={sem_pairs[0][2]}, "
          f"median={sem_pairs[len(sem_pairs)//2][2]}, "
          f"min={sem_pairs[-1][2]}")
    
    # Step 2: Check pair compatibility
    def pairs_are_unrelated(pair1, pair2):
        a, b = pair1[0], pair1[1]
        c, d = pair2[0], pair2[1]
        
        for x in [a, b]:
            for y in [c, d]:
                if only_shared_features[x, y] > 0:
                    return False
                if orthrelated_matrix[x, y] > 0:
                    return False
        return True
    
    # Step 3: Build compatibility matrix
    n_pairs = len(sem_pairs)
    print(f"Building compatibility matrix for {n_pairs} pairs...")
    
    compatible = np.zeros((n_pairs, n_pairs), dtype=bool)
    
    for i in range(n_pairs):
        for j in range(i+1, n_pairs):
            if pairs_are_unrelated(sem_pairs[i], sem_pairs[j]):
                compatible[i, j] = True
                compatible[j, i] = True
        
        if i % 500 == 0:
            print(f"  Processed {i}/{n_pairs} pairs...")
    
    # Step 4: Greedy selection - prioritize strongest pairs
    used_words = set()
    used_pairs = set()
    quads = []
    
    for pair_idx in range(n_pairs):
        if pair_idx in used_pairs:
            continue
            
        a, b, overlap_ab = sem_pairs[pair_idx]
        
        if a in used_words or b in used_words:
            continue
        
        # Find the strongest compatible partner
        compatible_indices = np.where(compatible[pair_idx])[0]
        
        best_partner = None
        best_overlap = -1
        
        for partner_idx in compatible_indices:
            if partner_idx in used_pairs:
                continue
            
            c, d, overlap_cd = sem_pairs[partner_idx]
            
            if c in used_words or d in used_words:
                continue
            
            if overlap_cd > best_overlap:
                best_partner = partner_idx
                best_overlap = overlap_cd
        
        if best_partner is not None:
            c, d, overlap_cd = sem_pairs[best_partner]
            
            quads.append({
                'words': (a, b, c, d),
                'sem_overlap_ab': overlap_ab,
                'sem_overlap_cd': overlap_cd
            })
            
            used_words.update([a, b, c, d])
            used_pairs.add(pair_idx)
            used_pairs.add(best_partner)
    
    print(f"\nFound {len(quads)} quads ({len(quads) * 4} words)")
    
    # Report
    all_overlaps = [q['sem_overlap_ab'] for q in quads] + [q['sem_overlap_cd'] for q in quads]
    print(f"Final semantic overlap stats:")
    print(f"  Mean: {np.mean(all_overlaps):.2f}")
    print(f"  Min: {np.min(all_overlaps)}")
    print(f"  Max: {np.max(all_overlaps)}")
    print(f"  Pairs with >1 shared feature: {sum(1 for o in all_overlaps if o > 1)}/{len(all_overlaps)}")
    
    return quads


def create_stimulus_lists(quads, lexicon):
    """From quads, create the three orderings."""
    standard = []
    sem_related = []
    unrelated = []
    
    for q in quads:
        a, b, c, d = q['words']
        
        standard.extend([a, b, c, d])
        sem_related.extend([b, a, d, c])
        unrelated.extend([c, d, a, b])
    
    return {
        'standard_idx': np.array(standard),
        'sem_related_idx': np.array(sem_related),
        'unrelated_idx': np.array(unrelated),
        'standard_words': [lexicon.words[i] for i in standard],
        'sem_related_words': [lexicon.words[i] for i in sem_related],
        'unrelated_words': [lexicon.words[i] for i in unrelated],
    }


# Usage:
# quads = find_counterbalanced_quads(lexicon, min_sem_overlap=1)
# stim_dict = create_stimulus_lists(quads, lexicon)