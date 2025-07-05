#!/usr/bin/env python3
"""
Convert JSON file to YAML file while preserving order.
"""

import json
import yaml
import argparse
import sys
from pathlib import Path


def load_json_with_order(json_file):
    """Load JSON file while preserving order using OrderedDict."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f, object_pairs_hook=dict)


def save_yaml_with_order(data, yaml_file):
    """Save data to YAML file while preserving order."""
    # Use default_flow_style=False for better readability
    # Use sort_keys=False to preserve order
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, 
                 allow_unicode=True, indent=2)


def convert_json_to_yaml(input_file, output_file=None):
    """Convert JSON file to YAML file."""
    try:
        # Load JSON data
        print(f"Loading JSON file: {input_file}")
        data = load_json_with_order(input_file)
        
        # Determine output file name if not provided
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.with_suffix('.yaml')
        
        # Save as YAML
        print(f"Converting to YAML: {output_file}")
        save_yaml_with_order(data, output_file)
        
        print(f"Conversion completed successfully!")
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{input_file}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON file to YAML file while preserving order",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.json
  %(prog)s data.json -o data.yaml
  %(prog)s data.json --output custom_name.yaml
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Input JSON file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        help='Output YAML file path (default: input_file.yaml)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    # Convert file
    convert_json_to_yaml(args.input_file, args.output_file)


if __name__ == "__main__":
    main() 