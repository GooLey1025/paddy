#!/usr/bin/env python3
"""
Quick TFRecord file explorer
Usage: python explore_tfr.py example.tfr [--max-records N] [--show-values]
"""

import tensorflow as tf
import sys
import numpy as np
import argparse
import os
from collections import defaultdict

def parse_feature_value(feature_value):
    """Parse a feature value and return its type and description"""
    if feature_value.HasField('bytes_list'):
        values = feature_value.bytes_list.value
        if len(values) == 1:
            # Try to decode as raw data to get shape info
            raw_data = values[0]
            return f"bytes(len={len(raw_data)})", raw_data
        else:
            return f"bytes_list(count={len(values)})", values
    elif feature_value.HasField('float_list'):
        values = feature_value.float_list.value
        if len(values) <= 10:
            return f"float_list({list(values)})", values
        else:
            return f"float_list(count={len(values)}, mean={np.mean(values):.3f})", values
    elif feature_value.HasField('int64_list'):
        values = feature_value.int64_list.value
        if len(values) <= 10:
            return f"int64_list({list(values)})", values
        else:
            return f"int64_list(count={len(values)}, mean={np.mean(values):.1f})", values
    else:
        return "unknown", None

def try_decode_raw_bytes(raw_bytes, common_dtypes=['float32', 'uint8', 'int32', 'float64']):
    """Try to decode raw bytes as different data types to infer structure"""
    results = []
    
    for dtype in common_dtypes:
        try:
            if dtype == 'float32':
                data = np.frombuffer(raw_bytes, dtype=np.float32)
            elif dtype == 'uint8':
                data = np.frombuffer(raw_bytes, dtype=np.uint8)
            elif dtype == 'int32':
                data = np.frombuffer(raw_bytes, dtype=np.int32)
            elif dtype == 'float64':
                data = np.frombuffer(raw_bytes, dtype=np.float64)
            
            # Try to infer reasonable shapes
            total_elements = len(data)
            possible_shapes = []
            
            # Common sequence lengths and depths
            common_lengths = [32768, 16384, 8192, 1024, 512, 256, 128, 64, 32, 23, 4]
            
            for length in common_lengths:
                if total_elements % length == 0:
                    depth = total_elements // length
                    if depth > 0 and depth <= 10000:  # Reasonable depth limit
                        possible_shapes.append((length, depth))
            
            # Single dimension
            possible_shapes.append((total_elements,))
            
            # Limit to most likely shapes
            best_shapes = possible_shapes[:3]
            
            results.append({
                'dtype': dtype,
                'total_elements': total_elements,
                'possible_shapes': best_shapes,
                'data_range': (float(np.min(data)), float(np.max(data))),
                'data_mean': float(np.mean(data))
            })
            
        except Exception as e:
            continue
    
    return results

def explore_tfrecord(filename, max_records=5, show_values=False, verbose=False):
    """Explore a single TFRecord file"""
    print(f"🔍 Exploring TFRecord: {filename}")
    
    # Check if file exists and get size
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return
    
    file_size = os.path.getsize(filename)
    if file_size > 1e9:
        size_str = f"{file_size/1e9:.1f}GB"
    elif file_size > 1e6:
        size_str = f"{file_size/1e6:.1f}MB"
    elif file_size > 1e3:
        size_str = f"{file_size/1e3:.1f}KB"
    else:
        size_str = f"{file_size}B"
    
    print(f"📏 File size: {size_str}")
    print("=" * 60)
    
    # Determine compression
    compression_type = ""
    if filename.endswith('.gz') or 'gz' in filename:
        compression_type = "GZIP"
    elif filename.endswith('.zlib') or 'zlib' in filename:
        compression_type = "ZLIB"
    else:
        # Try to auto-detect by attempting different compressions
        for comp in ["", "GZIP", "ZLIB"]:
            try:
                dataset = tf.data.TFRecordDataset(filename, compression_type=comp)
                for _ in dataset.take(1):
                    compression_type = comp if comp else "None"
                    break
                break
            except:
                continue
    
    print(f"🗜️  Compression: {compression_type if compression_type else 'None'}")
    
    try:
        # Create dataset
        dataset = tf.data.TFRecordDataset(filename, compression_type=compression_type)
        
        # Collect statistics
        feature_stats = defaultdict(lambda: {'count': 0, 'types': set(), 'examples': []})
        record_count = 0
        
        print(f"📄 Analyzing first {max_records} records...")
        print("=" * 60)
        
        for i, raw_record in enumerate(dataset.take(max_records)):
            record_count += 1
            print(f"\n📋 Record #{i+1}:")
            
            # Parse the example
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            # Analyze features
            for feature_name, feature_value in example.features.feature.items():
                feature_stats[feature_name]['count'] += 1
                
                type_desc, raw_data = parse_feature_value(feature_value)
                feature_stats[feature_name]['types'].add(type_desc.split('(')[0])
                
                print(f"  🔧 {feature_name}: {type_desc}")
                
                # Try to decode raw bytes and infer structure
                if raw_data is not None and isinstance(raw_data, bytes) and len(raw_data) > 4:
                    decode_results = try_decode_raw_bytes(raw_data)
                    if decode_results and verbose:
                        print(f"     🔍 Possible interpretations:")
                        for result in decode_results[:2]:  # Show top 2 interpretations
                            shapes_str = ", ".join([f"{s}" for s in result['possible_shapes'][:2]])
                            print(f"        - {result['dtype']}: shapes={shapes_str}, range=({result['data_range'][0]:.3f}, {result['data_range'][1]:.3f})")
                
                # Store example for summary
                if len(feature_stats[feature_name]['examples']) < 3:
                    feature_stats[feature_name]['examples'].append(type_desc)
                
                # Show values if requested and data is small
                if show_values and raw_data is not None:
                    if isinstance(raw_data, (list, tuple)) and len(raw_data) <= 5:
                        print(f"     └─ values: {raw_data}")
                    elif isinstance(raw_data, bytes) and len(raw_data) <= 20:
                        # Show as hex for small byte arrays
                        hex_str = raw_data.hex()[:40]
                        if len(hex_str) == 40:
                            hex_str += "..."
                        print(f"     └─ hex: {hex_str}")
        
        # Count total records (this might be slow for large files)
        print(f"\n📊 Counting total records...")
        try:
            total_count = sum(1 for _ in dataset)
            print(f"📈 Total records in file: {total_count}")
        except Exception as e:
            print(f"⚠️  Could not count total records: {e}")
            total_count = "unknown"
        
        # Feature summary
        print("=" * 60)
        print("📋 Feature Summary:")
        for feature_name, stats in feature_stats.items():
            types_str = ", ".join(stats['types'])
            print(f"  🔧 {feature_name}:")
            print(f"     - Type(s): {types_str}")
            print(f"     - Appears in: {stats['count']}/{record_count} records")
            if stats['examples']:
                print(f"     - Examples: {stats['examples'][0]}")
        
        print("=" * 60)
        print(f"✅ Analysis complete!")
        print(f"   Records analyzed: {record_count}")
        print(f"   Unique features: {len(feature_stats)}")
        print(f"   Total records: {total_count}")
        
    except Exception as e:
        print(f"❌ Error reading TFRecord: {e}")
        print(f"   This might be due to:")
        print(f"   - Wrong compression type")
        print(f"   - Corrupted file")
        print(f"   - Non-standard TFRecord format")

def explore_directory(directory, pattern="*.tfr", max_records=3):
    """Explore all TFRecord files in a directory"""
    import glob
    
    # Common TFRecord patterns
    patterns = [pattern, "*.tfrecord", "*.tfrecords", "*-*.tfr"]
    
    all_files = []
    for pat in patterns:
        files = glob.glob(os.path.join(directory, pat))
        all_files.extend(files)
    
    # Remove duplicates and sort
    all_files = sorted(list(set(all_files)))
    
    if not all_files:
        print(f"❌ No TFRecord files found in {directory}")
        print(f"   Searched patterns: {patterns}")
        return
    
    print(f"🔍 Found {len(all_files)} TFRecord files in {directory}")
    print("=" * 60)
    
    for i, filepath in enumerate(all_files):
        print(f"\n📁 File {i+1}/{len(all_files)}: {os.path.basename(filepath)}")
        explore_tfrecord(filepath, max_records=max_records, show_values=False, verbose=False)
        if i < len(all_files) - 1:
            print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Explore TFRecord file structure and contents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python explore_tfr.py data.tfr                           # Analyze single file
  python explore_tfr.py data.tfr --max-records 10         # Show first 10 records
  python explore_tfr.py data.tfr --show-values            # Show feature values
  python explore_tfr.py data.tfr -v                       # Verbose mode with data type inference
  python explore_tfr.py /path/to/tfrecords/ --directory   # Analyze all .tfr files in directory
        """
    )
    
    parser.add_argument('path', 
                       help='TFRecord file or directory to explore')
    parser.add_argument('-mr', '--max-records', 
                       type=int, 
                       default=5,
                       help='Maximum number of records to analyze in detail (default: 5)')
    parser.add_argument('-sv', '--show-values',
                       action='store_true',
                       help='Show actual feature values for small data')
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Show verbose information including data type inference')
    parser.add_argument('-d', '--directory',
                       action='store_true',
                       help='Treat path as directory and explore all TFRecord files')
    parser.add_argument('-p', '--pattern',
                       default='*.tfr',
                       help='File pattern for directory mode (default: *.tfr)')
    
    args = parser.parse_args()
    
    if args.directory or os.path.isdir(args.path):
        explore_directory(args.path, args.pattern, args.max_records)
    else:
        explore_tfrecord(args.path, args.max_records, args.show_values, args.verbose)

if __name__ == "__main__":
    main()
