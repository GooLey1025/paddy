#!/usr/bin/env python3
"""
Format metrics TSV files script
Extract content after 'se_eval_exps_' from directory name as prefix,
modify the first column of all_metrics.tsv files.
Required format: *_*_* (at least 2 underscores). Skip if already in correct format.
"""

import os
import re
import sys
from pathlib import Path


def extract_prefix_from_path(tsv_path):
    """Extract content after 'se_eval_exps_' from TSV file path as prefix"""
    path_str = str(tsv_path)
    
    # Search for se_eval_exps_ pattern
    match = re.search(r'se_eval_exps_([^/]+)', path_str)
    if match:
        return match.group(1)
    return None


def is_valid_format(text):
    """Check if text follows *_*_* format (at least 2 underscores)"""
    return len(text.split('_')) >= 3


def process_tsv_file(tsv_path, prefix):
    """Process single TSV file, modify first column content (skip header)"""
    try:
        # Read file
        with open(tsv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"Warning: {tsv_path} is empty")
            return False
        
        modified = False
        new_lines = []
        
        for i, line in enumerate(lines):
            line = line.rstrip('\n\r')
            
            if not line.strip():  # Empty line
                new_lines.append(line + '\n')
                continue
            
            # Split line
            columns = line.split('\t')
            if not columns:
                new_lines.append(line + '\n')
                continue
            
            # Skip header line (first line), keep it as is
            if i == 0:
                print(f"  Header: keeping '{columns[0]}' unchanged")
                new_lines.append(line + '\n')
                continue
            
            # For data rows, check first column
            original_first_col = columns[0]
            
            # If first column is already the prefix, don't modify
            if original_first_col == prefix:
                print(f"  Line {i+1}: '{original_first_col}' already matches prefix, skipping")
                new_lines.append(line + '\n')
            else:
                # Replace first column with prefix
                columns[0] = prefix
                new_line = '\t'.join(columns)
                new_lines.append(new_line + '\n')
                print(f"  Line {i+1}: '{original_first_col}' -> '{prefix}'")
                modified = True
        
        # Write back if modified
        if modified:
            with open(tsv_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"✓ Updated: {tsv_path}")
            return True
        else:
            print(f"- No changes needed: {tsv_path}")
            return False
            
    except Exception as e:
        print(f"Error: Failed to process file {tsv_path}: {e}")
        return False


def main():
    """Main function"""
    # Get current working directory
    current_dir = Path.cwd()
    print(f"Searching for all_metrics.tsv files in {current_dir}...")
    
    # Find all all_metrics.tsv files
    tsv_files = list(current_dir.rglob("**/all_metrics.tsv"))
    
    if not tsv_files:
        print("No all_metrics.tsv files found")
        return
    
    print(f"Found {len(tsv_files)} files:")
    for tsv_file in tsv_files:
        print(f"  {tsv_file}")
    
    print("\nStarting to process files...")
    
    processed_count = 0
    modified_count = 0
    
    for tsv_file in tsv_files:
        print(f"\nProcessing: {tsv_file}")
        
        # Extract prefix
        prefix = extract_prefix_from_path(tsv_file)
        
        if not prefix:
            print(f"  Warning: Cannot extract content after 'se_eval_exps_' from path, skipping")
            continue
        
        # Check prefix format
        if not is_valid_format(prefix):
            print(f"  Warning: Extracted prefix '{prefix}' doesn't follow *_*_* format, skipping")
            continue
        
        print(f"  Extracted prefix: '{prefix}'")
        
        # Process file
        was_modified = process_tsv_file(tsv_file, prefix)
        
        processed_count += 1
        if was_modified:
            modified_count += 1
    
    print(f"\nCompleted!")
    print(f"Processed {processed_count} files")
    print(f"Modified {modified_count} files")


if __name__ == "__main__":
    main()
