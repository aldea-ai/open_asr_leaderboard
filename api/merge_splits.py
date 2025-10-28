#!/usr/bin/env python3
"""
Merge split benchmark results into unified JSONL files per dataset.

Reads all *_RANGE_*.jsonl files in a directory, groups by dataset,
sorts by range, and combines them into single files with sequential indexing.
"""

import json
import re
import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple


def parse_range_from_filename(filename: str) -> Tuple[int, int]:
    """Extract (start, end) from RANGE_START-END in filename."""
    match = re.search(r'_RANGE_(\d+)-(\d+)', filename)
    if not match:
        raise ValueError(f"Could not parse range from {filename}")
    return int(match.group(1)), int(match.group(2))


def get_base_filename(filename: str) -> str:
    """Remove _RANGE_* suffix to get base dataset identifier."""
    return re.sub(r'_RANGE_\d+-\d+', '', filename)


def merge_splits(input_dir: str, output_dir: str = None, dry_run: bool = False):
    """
    Merge all split JSONL files in input_dir into unified files.
    
    Args:
        input_dir: Directory containing *_RANGE_*.jsonl files
        output_dir: Directory for merged outputs (defaults to input_dir)
        dry_run: If True, only print what would be done
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    
    # Find all split files
    split_files = list(input_path.glob("*_RANGE_*.jsonl"))
    
    if not split_files:
        print(f"No *_RANGE_*.jsonl files found in {input_dir}")
        return
    
    # Group files by base dataset name
    dataset_groups = defaultdict(list)
    for filepath in split_files:
        base_name = get_base_filename(filepath.name)
        try:
            range_start, range_end = parse_range_from_filename(filepath.name)
            dataset_groups[base_name].append({
                'path': filepath,
                'range_start': range_start,
                'range_end': range_end
            })
        except ValueError as e:
            print(f"Warning: {e}")
            continue
    
    # Process each dataset group
    for base_name, file_info_list in sorted(dataset_groups.items()):
        # Sort by range start
        file_info_list.sort(key=lambda x: x['range_start'])
        
        print(f"\n{'='*80}")
        print(f"Dataset: {base_name}")
        print(f"  Found {len(file_info_list)} split(s):")
        
        total_lines = 0
        for info in file_info_list:
            line_count = sum(1 for _ in open(info['path'], 'r'))
            total_lines += line_count
            print(f"    {info['path'].name:80s} ({line_count:6d} lines)")
        
        output_file = output_path / base_name
        
        if dry_run:
            print(f"  Would merge into: {output_file}")
            print(f"  Total lines: {total_lines}")
            continue
        
        # Merge files
        print(f"  Merging into: {output_file}")
        sample_index = 0
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for info in file_info_list:
                with open(info['path'], 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            # Renumber sample sequentially
                            data['audio_filepath'] = f"sample_{sample_index}"
                            out_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                            sample_index += 1
                        except json.JSONDecodeError as e:
                            print(f"    Warning: Failed to parse line in {info['path'].name}: {e}")
                            continue
        
        # Verify output
        output_lines = sum(1 for _ in open(output_file, 'r'))
        print(f"  ✓ Wrote {output_lines} lines (expected {total_lines})")
        
        if output_lines != total_lines:
            print(f"  ⚠ Warning: Line count mismatch!")


def main():
    parser = argparse.ArgumentParser(
        description="Merge split benchmark results into unified JSONL files"
    )
    parser.add_argument(
        "input_dir",
        nargs='?',
        default="./aldea",
        help="Directory containing *_RANGE_*.jsonl files (default: ./aldea)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for merged files (default: same as input)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Directory {args.input_dir} does not exist")
        return 1
    
    merge_splits(args.input_dir, args.output_dir, args.dry_run)
    print("\n" + "="*80)
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())

