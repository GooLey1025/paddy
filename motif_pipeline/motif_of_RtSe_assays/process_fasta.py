import re
import json
from collections import defaultdict

def validate_motif_position(sequence, motif_seq, position, sequence_length=32768):
    """
    Validate that the motif position is correctly mapped in the sequence.
    
    Args:
        sequence: The full gene sequence
        motif_seq: The motif sequence that was masked
        position: Position relative to sequence center (negative = upstream from center)
                 -220 means the rightmost base of motif is at position 220 upstream from center
        sequence_length: Expected sequence length (default 32768)
    
    Returns:
        tuple: (is_valid, message)
    """
    center = sequence_length // 2  # 16384 for 32768bp sequences
    
    if position < 0:
        # Negative position: upstream from center
        # Analysis shows the recorded position is 2 bases downstream from motif's actual right edge
        # So position -220 means motif's right edge is at position -222 relative to center
        motif_right_pos = center + position - 2  # e.g., 16384 + (-220) - 2 = 16162
        motif_left_pos = motif_right_pos - len(motif_seq) + 1
    else:
        # Positive position: downstream from center  
        # Assuming similar offset pattern for positive positions
        # The leftmost base of motif is at center + position - 2 (to be consistent)
        motif_left_pos = center + position - 2
        motif_right_pos = motif_left_pos + len(motif_seq) - 1
    
    # Check if positions are within sequence bounds
    if motif_left_pos < 0 or motif_right_pos >= len(sequence):
        return False, f"Motif position out of bounds: left={motif_left_pos}, right={motif_right_pos}, seq_len={len(sequence)}"
    
    # Extract the actual sequence at calculated positions
    actual_seq = sequence[motif_left_pos:motif_right_pos + 1]
    
    # Compare sequences (case-insensitive)
    if actual_seq.upper() == motif_seq.upper():
        return True, f"✓ Motif '{motif_seq}' correctly positioned at {position} (coordinates {motif_left_pos}-{motif_right_pos})"
    else:
        return False, f"✗ Motif mismatch at position {position}: expected '{motif_seq}', found '{actual_seq}' at coordinates {motif_left_pos}-{motif_right_pos}"

def process_fasta_file(input_file, output_all_genes, output_motif_dict):
    all_genes = {}
    motif_info = defaultdict(list)
    
    current_gene = None
    current_sequence = []
    
    print("Start processing fasta...")
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if not line:
                continue
                
            if line.startswith('>'):
                if current_gene and current_sequence:
                    sequence = ''.join(current_sequence)
                    if sequence:
                        all_genes[current_gene] = sequence
                
                current_sequence = []
                
                if '.Origin' in line:
                    current_gene = line[1:].replace('.Origin', '')
                    print(f"Processing gene: {current_gene}")
                    
                elif '|' in line and 'TFBS_Masked' in line:
                    parts = line[1:].split('|')
                    if len(parts) >= 4:
                        gene_id = parts[0]
                        motif_full = parts[1]
                        masked_seq = parts[2]
                        position = int(parts[3])
                        
                        if '.TFBS_Masked' in motif_full:
                            motif_id_name = motif_full.replace('.TFBS_Masked', '')
                            if '_' in motif_id_name:
                                motif_id, motif_name = motif_id_name.split('_', 1)
                            else:
                                motif_id = motif_id_name
                                motif_name = motif_id_name
                        else:
                            motif_id = motif_full
                            motif_name = motif_full
                        
                        # Calculate absolute coordinates for 32768bp sequence (0-based)
                        center = 32768 // 2  # 16384
                        if position < 0:
                            # Negative position: upstream from center
                            # Position -220 means motif's right edge is at center + position - 2
                            motif_right_abs = center + position - 2  # 0-based coordinates
                            motif_left_abs = motif_right_abs - len(masked_seq) + 1
                        else:
                            # Positive position: downstream from center
                            motif_left_abs = center + position - 2  # 0-based coordinates
                            motif_right_abs = motif_left_abs + len(masked_seq) - 1
                        
                        # Calculate normalized relative position (like list indexing)
                        # -1 = left center, -2 = 2nd position left of center, etc.
                        normalized_right_pos = motif_right_abs - center  # relative to center
                        normalized_left_pos = motif_left_abs - center
                        
                        motif_info[gene_id].append({
                            'motif_id': motif_id,
                            'motif_name': motif_name,
                            'masked_sequence': masked_seq,
                            'position': position,  # original position from source
                            'absolute_start': motif_left_abs,
                            'absolute_end': motif_right_abs,
                            'normalized_start': normalized_left_pos,
                            'normalized_end': normalized_right_pos
                        })
                        
                        print(f"  - Found motif: {motif_id} ({motif_name}) at position {position}")
                    
                    current_gene = None
                else:
                    current_gene = None
            else:
                if current_gene:
                    current_sequence.append(line)
        
    if current_gene and current_sequence:
        sequence = ''.join(current_sequence)
        if sequence:
            all_genes[current_gene] = sequence
    
    print(f"\nWriting {len(all_genes)} genes to {output_all_genes}")
    with open(output_all_genes, 'w') as f:
        for gene_id, sequence in all_genes.items():
            f.write(f">{gene_id}\n")
            f.write(f"{sequence}\n")
    
    print(f"Writing motif info to {output_motif_dict}")
    motif_dict = dict(motif_info)
    with open(output_motif_dict, 'w') as f:
        json.dump(motif_dict, f, indent=2, ensure_ascii=False)
    
    # Validate motif positions after all sequences are loaded
    print(f"\nValidating motif positions...")
    validation_errors = 0
    validation_success = 0
    
    for gene_id, motifs in motif_info.items():
        if gene_id in all_genes:
            gene_sequence = all_genes[gene_id]
            print(f"\nValidating {gene_id} (sequence length: {len(gene_sequence)}):")
            
            for motif in motifs:
                is_valid, msg = validate_motif_position(
                    gene_sequence, 
                    motif['masked_sequence'], 
                    motif['position']
                )
                print(f"  {msg}")
                print(f"    Absolute coordinates: {motif['absolute_start']}-{motif['absolute_end']} (0-based)")
                print(f"    Normalized positions: {motif['normalized_start']} to {motif['normalized_end']} (relative to center)")
                
                if is_valid:
                    validation_success += 1
                else:
                    validation_errors += 1
                    # Assert to catch validation failures
                    assert is_valid, f"Motif position validation failed for {gene_id}: {msg}"
        else:
            print(f"Warning: No sequence found for gene {gene_id}")
    
    print(f"\nValidation Summary:")
    print(f"- Successful validations: {validation_success}")
    print(f"- Failed validations: {validation_errors}")
    
    print(f"\nProcessing completed!")
    print(f"- Total genes processed: {len(all_genes)}")
    print(f"- Total motifs found: {sum(len(motifs) for motifs in motif_info.values())}")
    
    return all_genes, motif_dict

if __name__ == "__main__":
    input_file = "Genes_with_Nmask.seq.fa"
    output_all_genes = "all_genes.fa"
    output_motif_dict = "motif_info.json"
    
    all_genes, motif_dict = process_fasta_file(input_file, output_all_genes, output_motif_dict)
