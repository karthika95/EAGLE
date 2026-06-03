"""
Build word-group-aware SAM from processed Hindi data.
This script creates a SAM that respects word group boundaries marked in the data.
"""
import os
import sys
import json
import argparse
import pickle
from typing import List, Tuple
from tqdm import tqdm

# Add parent directory to path to import samd modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samd.wordgroup_sam import WordGroupAwareSAM


def load_processed_hindi_data(input_file: str) -> List[dict]:
    """
    Load processed Hindi data from JSON file.
    
    Args:
        input_file: Path to processed_nios.jsonl (JSON format)
        
    Returns:
        List of data dictionaries
    """
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries")
    
    # Validate data structure
    for i, entry in enumerate(data[:5]):
        if 'tokens' not in entry or 'word_group_boundaries' not in entry:
            print(f"Warning: Entry {i} missing required keys")
            print(f"Keys found: {entry.keys()}")
    
    return data


def dump_sam(path: str, sam: WordGroupAwareSAM):
    """Save SAM to pickle file."""
    print(f"Saving SAM to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(sam, f)
    print("SAM saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Build word-group-aware SAM from processed Hindi data"
    )
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='/nfs/kundeshwar/pranav-shinde/download/processed_nios.jsonl',
        help='Path to processed Hindi data (JSON format)'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='downloads/sam_hindi_wordgroup.pkl',
        help='Path to save the SAM pickle file'
    )
    parser.add_argument(
        '--n_predicts', 
        type=int, 
        default=15,
        help='Maximum number of tokens to predict (will stop at word boundaries)'
    )
    parser.add_argument(
        '--eos_token', 
        type=int, 
        default=2,  # Common EOS token ID for Llama-based models
        help='End of sequence token ID'
    )
    
    args = parser.parse_args()
    
    # Load processed data
    hindi_data = load_processed_hindi_data(args.input_file)
    
    print(f"\nBuilding word-group-aware SAM with n_predicts={args.n_predicts}...")
    print(f"Total sequences: {len(hindi_data)}")
    
    # Build SAM
    sam = WordGroupAwareSAM.build(
        batch_data=hindi_data,
        eos_token=args.eos_token,
        n_predicts=args.n_predicts,
        verbose=True
    )
    
    # Print statistics
    print(f"\nSAM Statistics:")
    print(f"  Total states: {len(sam.states)}")
    print(f"  Total tokens: {len(sam.input_ids)}")
    print(f"  Word boundaries: {sum(sam.word_boundaries)}")
    print(f"  Max length: {sam.max_length}")
    
    # Save SAM
    dump_sam(args.output_path, sam)
    
    print(f"\n✓ Successfully created word-group-aware SAM!")
    print(f"  Saved to: {args.output_path}")


if __name__ == '__main__':
    main()