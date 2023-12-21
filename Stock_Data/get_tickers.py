"""

This script connects to the NASDAQ Trader FTP server
and updates the 'nasdaqlisted.txt' and 'otherlisted.txt' files.
It then extracts the stock symbols from these files, 
and writes these symbols into two separate Python files ('others.py' and 'nasdaq.py') as lists.

"""
import ftplib


def get_symbols(txt_file_path):
    symbols = []

    with open(txt_file_path, 'r') as txt_file:
        lines = txt_file.readlines()
        
        # Assuming the first line contains the headers
        headers = [header.strip() for header in lines[0].split('|')]
        
        # Find the index of the "Symbol" column
        if "Symbol" in headers:
            symbol_index = headers.index("Symbol")
        elif "ACT Symbol" in headers:
            symbol_index = headers.index("ACT Symbol")
        else:
            print("Error: 'Symbol' or 'ACT Symbol' column not found.")
            return []

        # Extract symbols from the subsequent lines
        for line in lines[1:]:
            values = [value.strip() for value in line.split('|')]
            symbols.append(values[symbol_index])

    return symbols

def write_list_to_py(filename, lst_name, lst):
    with open(filename, 'w') as f:
        f.write(f"{lst_name} = {lst}")

ftp = ftplib.FTP("ftp.nasdaqtrader.com")
ftp.login("anonymous", "")
ftp.cwd("SymbolDirectory")
files_to_download = ["nasdaqlisted.txt", "otherlisted.txt"]
for file in files_to_download:
  with open(file, "wb") as f:
    ftp.retrbinary(f"RETR {file}", f.write)
ftp.quit()

others = get_symbols('otherlisted.txt')
nasdaq = get_symbols('nasdaqlisted.txt')

write_list_to_py('others.py', 'others', others)
write_list_to_py('nasdaq.py', 'nasdaq', nasdaq)