#!/usr/bin/env python3
"""Convert the NF-UQ-NIDS-v2 CSV from the University of Queensland to parquet.

The CSV is large (tens of millions of rows), so it is streamed in blocks instead of
loaded at once. Usage:

    python scripts/csv_to_parquet.py NF-UQ-NIDS-v2.csv data/raw/NF-UQ-NIDS-v2.parquet
"""
import os
import sys

import pyarrow.csv as pacsv
import pyarrow.parquet as pq


def main(src, dst):
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    reader = pacsv.open_csv(src, read_options=pacsv.ReadOptions(block_size=1 << 26))
    writer, rows = None, 0
    for batch in reader:
        if writer is None:
            writer = pq.ParquetWriter(dst, batch.schema)
        writer.write_batch(batch)
        rows += batch.num_rows
    if writer is not None:
        writer.close()
    print(f"wrote {rows:,} rows to {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
