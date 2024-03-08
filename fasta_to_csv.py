#!/usr/bin/env python3

import sys
import pandas as pd
from pathlib import Path
from Bio.SeqIO.FastaIO import SimpleFastaParser

FASTA = sys.argv[1]


def fas_to_csv(in_fasta: str) -> None:
    in_fasta    = Path(in_fasta)
    basename    = in_fasta.stem
    csv_out     = f"{basename}.csv"
    # Initialize lists to store ids, labels and sequences.
    fids = []
    fsqs = []
    labs = []
    with open(in_fasta,"r") as fasta, open(csv_out, "w") as fout:
        # Read fasta
        for fid, fsq in SimpleFastaParser(fasta):
            fid, lab = fid.split('#')
            fids.append(fid)
            fsqs.append(fsq)
            labs.append(lab)
        # Write to CSV file
        pd.DataFrame({'id':         fids,
                      'label':      labs,
                      'sequence':   fsqs}) \
        .to_csv(fout, index=False)


if __name__ == '__main__':
    print(f"### Reading file: {FASTA}")
    fas_to_csv(FASTA)
    print('Done!')
