#!/usr/bin/env python3
"""
Fix Unicode escape sequences in JSONL files.

This script reads a JSONL file with Unicode escapes (like \u0905) and
converts them to actual readable characters (like अ).

Usage:
    python scripts/fix_unicode_in_jsonl.py <input_file> [output_file]

If output_file is not specified, creates <input_file>.fixed

Example:
    /data/pranav_shinde/pranav/SAM-Decoding/evaluation/data/mt_bench/model_answer/airavata-samd-wordgroup.jsonl
"""

import json
import sys
import os
from pathlib import Path


def fix_unicode_in_jsonl(input_path, output_path=None):
    """
    Read JSONL file and rewrite with proper Unicode characters.
    
    Args:
        input_path: Path to input JSONL file with Unicode escapes
        output_path: Path to output file (default: input_path + '.fixed')
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return False
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.fixed{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    # Create backup
    backup_path = input_path.parent / f"{input_path.stem}.backup{input_path.suffix}"
    if not backup_path.exists():
        print(f"Creating backup: {backup_path}")
        import shutil
        shutil.copy2(input_path, backup_path)
    
    line_count = 0
    error_count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as fin:
            with open(output_path, 'w', encoding='utf-8') as fout:
                for line_num, line in enumerate(fin, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parse JSON (this automatically decodes Unicode escapes)
                        data = json.loads(line)
                        
                        # Write back with ensure_ascii=False to preserve Unicode characters
                        fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                        line_count += 1
                        
                        if line_count % 10 == 0:
                            print(f"Processed {line_count} lines...", end='\r')
                    
                    except json.JSONDecodeError as e:
                        print(f"\nError decoding line {line_num}: {e}")
                        error_count += 1
                        # Write original line to not lose data
                        fout.write(line + '\n')
        
        print(f"\nDone! Processed {line_count} lines")
        if error_count > 0:
            print(f"Errors encountered: {error_count}")
        
        # Show sample of fixed output
        print("\n" + "="*80)
        print("Sample of fixed output (first 3 turns from first question):")
        print("="*80)
        with open(output_path, 'r', encoding='utf-8') as f:
            first_line = json.loads(f.readline())
            if 'choices' in first_line and len(first_line['choices']) > 0:
                turns = first_line['choices'][0].get('turns', [])
                for i, turn in enumerate(turns[:3], 1):
                    print(f"\nTurn {i}:")
                    print(turn[:200] + ("..." if len(turn) > 200 else ""))
        print("="*80)
        
        return True
    
    except Exception as e:
        print(f"\nError processing file: {e}")
        import traceback
        traceback.print_exc()
        return False


def replace_original(input_path):
    """Replace original file with fixed version"""
    input_path = Path(input_path)
    fixed_path = input_path.parent / f"{input_path.stem}.fixed{input_path.suffix}"
    
    if not fixed_path.exists():
        print(f"Error: Fixed file not found: {fixed_path}")
        return False
    
    print(f"\nReplacing original file with fixed version...")
    print(f"Original: {input_path}")
    print(f"Fixed:    {fixed_path}")
    
    # Rename original to .old if backup doesn't exist
    backup_path = input_path.parent / f"{input_path.stem}.backup{input_path.suffix}"
    if not backup_path.exists():
        print(f"Moving original to: {backup_path}")
        input_path.rename(backup_path)
    else:
        print(f"Removing original (backup exists at {backup_path})")
        input_path.unlink()
    
    # Rename fixed to original name
    print(f"Renaming fixed file to: {input_path}")
    fixed_path.rename(input_path)
    
    print("✓ Done! Original file has been replaced.")
    print(f"  Backup saved at: {backup_path}")
    
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Please provide input file path")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Fix the file
    success = fix_unicode_in_jsonl(input_file, output_file)
    
    if not success:
        sys.exit(1)
    
    # Ask if user wants to replace original
    if output_file is None:  # Only ask if we created a .fixed file
        print("\n" + "="*80)
        response = input("\nReplace original file with fixed version? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            replace_original(input_file)
        else:
            fixed_path = Path(input_file).parent / f"{Path(input_file).stem}.fixed{Path(input_file).suffix}"
            print(f"\nOriginal file unchanged. Fixed version at: {fixed_path}")
            print(f"To replace manually, run:")
            print(f"  mv {input_file} {input_file}.old")
            print(f"  mv {fixed_path} {input_file}")


if __name__ == "__main__":
    main()
