import pandas as pd
import argparse

def extract_label_to_csv(xlsx_file_path, output_csv_file_path, start_column=8):
    df = pd.read_excel(xlsx_file_path)

    extracted_columns = df.iloc[:, start_column:]
    extracted_columns.to_csv(output_csv_file_path, sep='\t', index=False)
    print(f"Extracted {extracted_columns.shape[1]} columns from {xlsx_file_path} to {output_csv_file_path}")

    return None


def main():
    parser = argparse.ArgumentParser(description=
                                     "Extract label from xlsx file to csv file, "
                                     "e.g. python extract_label.py "
                                     "-x Nip_ATGsite_UD16K_Bed.xlsx "
                                     "-o 2_P8_Nip8_23tissues.Exp.csv "
                                     "-s 8"
                                     )
    parser.add_argument("-x", "--xlsx_file_path", type=str, required=True)
    parser.add_argument("-o", "--output_csv_file_path", type=str, required=True)
    parser.add_argument("-s", "--start_column", type=int, default=8)
    options = parser.parse_args()

    xlsx_file_path = options.xlsx_file_path
    output_csv_file_path = options.output_csv_file_path
    start_column = options.start_column

    extract_label_to_csv(xlsx_file_path, output_csv_file_path, start_column)

if __name__ == "__main__":
    main()
