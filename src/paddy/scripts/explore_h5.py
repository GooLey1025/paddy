#!/usr/bin/env python3
"""
Quick H5 file structure explorer
Usage: python explore_h5.py example.h5 [--max-depth N]
"""

import h5py
import sys
import numpy as np
import argparse

def display_attributes(item, prefix=""):
    """Display attributes of an H5 item (group or dataset)"""
    if len(item.attrs) == 0:
        return
    
    for attr_name, attr_value in item.attrs.items():
        # Format attribute value based on type
        if isinstance(attr_value, np.ndarray):
            if attr_value.size <= 5:
                value_str = str(attr_value.tolist())
            else:
                value_str = f"array{attr_value.shape} ({attr_value.dtype})"
        elif isinstance(attr_value, bytes):
            try:
                value_str = attr_value.decode('utf-8')
            except:
                value_str = f"bytes({len(attr_value)})"
        elif isinstance(attr_value, str) and len(attr_value) > 50:
            value_str = attr_value[:47] + "..."
        else:
            value_str = str(attr_value)
        
        print(f"{prefix}  🔧 {attr_name}: {value_str}")

def explore_h5_structure(h5_file, prefix="", max_depth=10, current_depth=1, stats=None):
    """Recursively explore H5 file structure
    
    Args:
        current_depth: Current depth level (1 = top level, 2 = second level, etc.)
        max_depth: Maximum depth to explore (1 = only top level)
    """
    if stats is None:
        stats = {'groups': 0, 'datasets': 0, 'total_size': 0}
    
    for key in h5_file.keys():
        item = h5_file[key]
        
        if isinstance(item, h5py.Group):
            stats['groups'] += 1
            print(f"{prefix}📁 {key}/")
            # Display attributes for this group
            display_attributes(item, prefix)
            # Only recurse if current depth is less than max_depth
            if current_depth < max_depth:
                explore_h5_structure(item, prefix + "  ", max_depth, current_depth + 1, stats)
            elif len(list(item.keys())) > 0:
                # Show that there's more content but we've reached the limit
                print(f"{prefix}  ... (max depth {max_depth} reached)")
        
        elif isinstance(item, h5py.Dataset):
            stats['datasets'] += 1
            shape_str = f"shape={item.shape}"
            dtype_str = f"dtype={item.dtype}"
            
            # Add size info for large arrays
            size = np.prod(item.shape) if item.shape else 1
            stats['total_size'] += size
            
            print(f"{prefix}📄 {key} - {shape_str}, {dtype_str}")
            # Display attributes for this dataset
            display_attributes(item, prefix)
            
            # Show first few values for small arrays
            if size <= 10 and item.dtype.kind in ['i', 'f', 'U', 'S']:
                try:
                    values = item[()]
                    if isinstance(values, np.ndarray) and values.size <= 10:
                        print(f"{prefix}   └─ values: {values}")
                    elif not isinstance(values, np.ndarray):
                        print(f"{prefix}   └─ value: {values}")
                except:
                    pass
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description='Explore H5 file structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python explore_h5.py data.h5                    # Explore with default depth (10)
  python explore_h5.py data.h5 --max-depth 1     # Show only top-level (like du --max-depth 1)
  python explore_h5.py data.h5 -md 2             # Show top-level and one level down
  python explore_h5.py data.h5 -md 0             # Unlimited depth
        """
    )
    
    parser.add_argument('filename', 
                       help='H5 file to explore')
    parser.add_argument('-md', '--max-depth', 
                       type=int, 
                       default=10,
                       help='Maximum depth to explore (default: 10, use 0 for unlimited)')
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Show more detailed information')
    
    args = parser.parse_args()
    
    # Handle unlimited depth
    max_depth = float('inf') if args.max_depth == 0 else args.max_depth
    
    try:
        with h5py.File(args.filename, 'r') as f:
            print(f"🔍 Exploring H5 file: {args.filename}")
            if args.max_depth == 0:
                print("📏 Max depth: unlimited")
            else:
                print(f"📏 Max depth: {args.max_depth}")
            print("=" * 50)
            
            # Display file-level attributes first
            if len(f.attrs) > 0:
                print("📋 File attributes:")
                display_attributes(f, "")
                print("=" * 50)
            
            stats = explore_h5_structure(f, max_depth=max_depth)
            print("=" * 50)
            
            # Display summary statistics
            total_size = stats['total_size']
            if total_size > 1e9:
                size_str = f"{total_size/1e9:.1f}B"
            elif total_size > 1e6:
                size_str = f"{total_size/1e6:.1f}M"
            elif total_size > 1e3:
                size_str = f"{total_size/1e3:.1f}K"
            else:
                size_str = f"{total_size}"
            
            print(f"📊 Summary: {stats['groups']} groups, {stats['datasets']} datasets, {size_str} total elements")
            print("✅ Done!")
            
    except FileNotFoundError:
        print(f"❌ Error: File '{args.filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
