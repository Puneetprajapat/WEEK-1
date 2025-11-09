import pandas as pd
import matplotlib
# Use a non-interactive backend so the script doesn't block on plt.show()
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parent
INFILE = BASE / 'household_power_consumption.txt'
OUTFILE = BASE / 'household_power_consumption_clean.csv'

# 1) Read with proper NA handling (the dataset uses '?' for missing values)
df = pd.read_csv(INFILE, sep=';', na_values='?', low_memory=False)

# Quick overview
print('Initial shape:', df.shape)
print('Columns:', list(df.columns))

# 2) Parse datetime: handle both 'Date'/'Time' or lowercase 'date' if present
if 'Date' in df.columns and 'Time' in df.columns:
	df['Datetime'] = pd.to_datetime(df['Date'].str.strip() + ' ' + df['Time'].str.strip(), dayfirst=True, errors='coerce')
elif 'date' in df.columns and 'time' in df.columns:
	df['Datetime'] = pd.to_datetime(df['date'].str.strip() + ' ' + df['time'].str.strip(), dayfirst=True, errors='coerce')
else:
	# If no separate date/time, try to parse a single datetime-like column
	possible = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'datetime' in c.lower()]
	if possible:
		df['Datetime'] = pd.to_datetime(df[possible[0]], dayfirst=True, errors='coerce')
	else:
		df['Datetime'] = pd.NaT

# 3) Convert all non-datetime, non-string columns to numeric where appropriate
# Identify numeric candidate columns (exclude Date/Time/Text columns)
exclude = set(['Date', 'Time', 'date', 'time', 'Datetime'])
numeric_cols = [c for c in df.columns if c not in exclude]

for col in numeric_cols:
	# coerce errors -> NaN for bad parsing (e.g. '?')
	df[col] = pd.to_numeric(df[col], errors='coerce')

# 4) Drop duplicates
before_dup = df.shape[0]
df = df.drop_duplicates()
print(f'dropped {before_dup - df.shape[0]} duplicate rows')

# 5) Handle missing values
# - Drop rows without a valid Datetime (essential)
before = df.shape[0]
df = df.dropna(subset=['Datetime'])
print(f'dropped {before - df.shape[0]} rows with invalid Datetime')

# For numeric columns, fill missing values with median (robust to outliers)
num_cols = df.select_dtypes(include=['number']).columns.tolist()
if num_cols:
	medians = df[num_cols].median()
	df[num_cols] = df[num_cols].fillna(medians)

# 6) Final info and quick stats
print('\nAfter cleaning:')
print(df.info())
print(df[num_cols].describe().T)

# 7) Optional: small visualization on the main active power column if present
plot_col = None
for candidate in ['Global_active_power', 'global_active_power', 'Global_active_power(kW)', 'energy_consumption', 'Energy']:
	if candidate in df.columns:
		plot_col = candidate
		break

if plot_col and df[plot_col].notna().any():
	try:
		plt.figure(figsize=(6, 3))
		plt.boxplot(df[plot_col].dropna())
		plt.title(f'Boxplot of {plot_col}')
		plt.tight_layout()
		# Save the plot instead of showing it (non-interactive)
		plot_path = BASE / f'{plot_col}_boxplot.png'
		plt.savefig(plot_path)
		print(f'Boxplot saved to: {plot_path}')
	except Exception as e:
		print('Plot failed:', e)

import argparse


def write_csv_safe(df, outpath: Path, full: bool = False, sample_n: int = 500000, chunk_size: int = 100000):
	"""Write CSV safely:
	- By default writes a sample (first sample_n rows) for quick validation.
	- If full=True, writes the full dataframe in chunks to avoid long blocking calls.
	"""
	if not full:
		sample = df.head(sample_n)
		sample.to_csv(outpath.with_name(outpath.stem + '_sample' + outpath.suffix), index=False)
		print(f'Sample ({sample_n} rows) written to: {outpath.with_name(outpath.stem + "_sample" + outpath.suffix)}')
		return

	# Full write: write in chunks to give progress updates and avoid a single huge conversion call
	total = df.shape[0]
	print(f'Writing full CSV in chunks (total rows: {total}, chunk_size: {chunk_size})')
	written = 0
	# Ensure output file does not exist
	if outpath.exists():
		outpath.unlink()

	for start in range(0, total, chunk_size):
		end = min(start + chunk_size, total)
		chunk = df.iloc[start:end]
		header = (start == 0)
		chunk.to_csv(outpath, mode='a', index=False, header=header)
		written += len(chunk)
		print(f'  wrote rows {start}..{end-1}  ({written}/{total})')

	print(f'Full cleaned file written to: {outpath}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Clean household power dataset and write CSV.')
	parser.add_argument('--full', action='store_true', help='Write the full cleaned CSV (may take a long time).')
	parser.add_argument('--sample', type=int, default=500000, help='Number of rows to write for sample output (default 500000).')
	parser.add_argument('--chunk-size', type=int, default=100000, help='Chunk size when writing full CSV (default 100000).')
	args = parser.parse_args()

	write_csv_safe(df, OUTFILE, full=args.full, sample_n=args.sample, chunk_size=args.chunk_size)

	# If a small boxplot was created, print its path (plot saving is optional and already handled above)
	if plot_col:
		plot_path = BASE / f'{plot_col}_boxplot.png'
		if plot_path.exists():
			print(f'Boxplot file: {plot_path}')
