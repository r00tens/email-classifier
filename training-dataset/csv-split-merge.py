import argparse
import logging
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# Ustawienie logowania / Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def split_data(input_file, test_size=0.2, random_state=42):
    """
    Podział pliku CSV na zestawy treningowy i testowy.

    Funkcja wczytuje plik CSV, sprawdza, czy plik istnieje i nie jest pusty,
    a następnie dzieli dane na dwa zestawy: treningowy i testowy. Użytkownik
    może określić wielkość zestawu testowego oraz wartość random_state
    dla powtarzalności podziału. Zestawy są zapisywane do osobnych plików CSV.

    Splits a CSV file into training and test sets.

    The function loads a CSV file, checks if the file exists and is not empty,
    then splits the data into training and test sets. The user can define the
    test set size and the random_state for reproducibility. The sets are saved
    into separate CSV files.
    """
    if not os.path.isfile(input_file):
        logging.error(f"The file '{input_file}' does not exist.")
        sys.exit(1)

    try:
        data = pd.read_csv(input_file)
    except pd.errors.EmptyDataError:
        logging.error(f"The file '{input_file}' is empty or not a valid CSV file.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading the file '{input_file}': {e}")
        sys.exit(1)

    if data.empty:
        logging.error(f"The file '{input_file}' contains no data.")
        sys.exit(1)

    try:
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
    except ValueError as e:
        logging.error(f"Error splitting the data: {e}")
        sys.exit(1)

    file_name, file_extension = os.path.splitext(input_file)
    train_file = f'{file_name}-train{file_extension}'
    test_file = f'{file_name}-test{file_extension}'

    try:
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        logging.info(f"Training data saved to: {train_file}")
        logging.info(f"Test data saved to: {test_file}")
    except Exception as e:
        logging.error(f"Error saving the split files: {e}")
        sys.exit(1)

    return train_data, test_data


def merge_csv(files, output_file):
    """
    Łączenie wielu plików CSV w jeden.

    Funkcja wczytuje listę plików CSV i sprawdza, czy każdy z nich istnieje.
    Jeśli pliki istnieją, są wczytywane do DataFrame, a następnie łączone w jeden
    duży DataFrame. Zestaw wynikowy jest zapisywany do nowego pliku CSV.

    Merges multiple CSV files into one.

    The function loads a list of CSV files and checks if each file exists.
    If the files exist, they are loaded into a DataFrame and then merged
    into one large DataFrame. The resulting dataset is saved to a new CSV file.
    """
    df_list = []

    for file in files:
        if os.path.exists(file):
            logging.info(f"Loading file: {file}")
            df = pd.read_csv(file)
            df_list.append(df)
        else:
            logging.error(f"The file {file} does not exist!")

    if not df_list:
        logging.error("No files to merge.")
        sys.exit(1)

    merged_df = pd.concat(df_list, ignore_index=True)

    try:
        merged_df.to_csv(output_file, index=False)
        logging.info(f"Files have been merged and saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving the merged file: {e}")
        sys.exit(1)


def main():
    """
    Główna funkcja obsługująca argumenty z linii poleceń.

    Program obsługuje dwie operacje: podział danych (split) i łączenie plików CSV (merge).
    Na podstawie wyboru użytkownika program wywołuje odpowiednie funkcje do obsługi tych operacji.
    Użytkownik może określić plik do podziału, jego rozmiar testowy oraz pliki do połączenia.

    Main function handling command-line arguments.

    The program supports two operations: data splitting (split) and CSV file merging (merge).
    Based on the user's choice, the program calls the appropriate functions to handle these
    operations. The user can specify the file to split, the test set size, and the files to merge.
    """
    parser = argparse.ArgumentParser(description="Program to split data and merge CSV files.")

    subparsers = parser.add_subparsers(dest='command', help='Choose an operation: split or merge')

    split_parser = subparsers.add_parser('split', help='Split a CSV file into training and test sets')
    split_parser.add_argument('input_file', type=str, help='Path to the CSV file containing data')
    split_parser.add_argument('--test_size', type=float, default=0.2,
                              help='Percentage of data for the test set (default is 20%%)')
    split_parser.add_argument('--random_state', type=int, default=42,
                              help='Random state for reproducibility (default is 42)')

    merge_parser = subparsers.add_parser('merge', help='Merge several CSV files into one')
    merge_parser.add_argument('files', nargs='+', help='Paths to the CSV files to merge.')
    merge_parser.add_argument('--output', type=str, default='merged.csv',
                              help='Output file path (default is merged.csv).')

    args = parser.parse_args()

    if args.command == 'split':
        split_data(args.input_file, args.test_size, args.random_state)
    elif args.command == 'merge':
        merge_csv(args.files, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
