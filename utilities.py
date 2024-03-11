# setup logging
import logger
log = logger.get_logger(__name__)

import glob
import pandas as pd
from pathlib import Path

def parquet_to_excel(glob_pattern: str) -> None:
    '''
    Converts parquet files to excel. The resulting files will be placed in the same folder by only changing the
    extension from .parquet to .xlsx
    :param glob_pattern: glob pattern to match, e.g. r'output/reference_substance_data/reference_substance_data_part*.parquet'
    :return: None
    '''
    files = glob.glob(glob_pattern)
    for file in files:
        # read parquet file
        file = Path(file)
        tmp = pd.read_parquet(file)
        fsize = file.stat().st_size
        log.info(f'read parquet file {file.name} with {len(tmp)} rows ({fsize} bytes)')
        # store excel file
        file = file.parent / Path(file.stem + '.xlsx')
        tmp.to_excel(file)
        fsize = file.stat().st_size
        log.info(f'created excel file {file.name} with {len(tmp)} rows ({fsize} bytes)')


def chunker(seq, size):
    '''
    Splits a sequence into chunks

    :param seq: sequence to split into chunks
    :param size: chunk size
    :return: a generators with the chunks
    '''
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def create_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]