import logger
log = logger.get_logger(__name__)

import pandas as pd
import requests
from dotenv import load_dotenv
import os
from itertools import chain
from tqdm import tqdm

# load the environment variables
load_dotenv()

def retrieve_structure(identifiers: list[str], batch_size: int=50) -> pd.DataFrame:
    '''
    Retrieve the DSSTox structures for a list of identifiers. The DSStox API key needs
     to be set in the environment variable DSSTOX_KEY. The function returns as many
     rows as the number of input identifiers, i.e. it returns a row even if the identifier
     could not be resolved. However, the input identifiers are made unique before the retrieval and null values are discarded.
    :param identifiers: list of identifiers to retrieve the structure for
    :param batch_size: batch size for the retrieval, default is 50
    :return: pandas dataframe with the retrieved structures; the original identifiers are kept in the 'identifier' column
    '''

    # set the headers
    headers = {
        "accept": "application/json",
        "x-api-key": os.getenv('DSSTOX_KEY')
    }

    # keep unique, non-null identifiers
    identifiers = list({identifier for identifier in identifiers if identifier})

    # retrieve the structures
    dsstox_data = []
    for i in tqdm(range(0, len(identifiers), batch_size)):
        try:
            # we attempt to run in a batch, but we raise an exception if the number of
            # responses does not equal the number of input identifiers
            identifiers_batch = identifiers[i:i + batch_size]
            ccd = f"https://api-ccte.epa.gov/chemical/search/equal/"
            response = requests.post(ccd, headers=headers, data='\n'.join(identifiers_batch))
            response = pd.DataFrame.from_records(response.json())
            if len(response) == len(identifiers_batch):
                dsstox_data.append(response.assign(identifier=identifiers_batch))
            else:
                ex = Exception(f'batch {i}-{i + batch_size-1} returned {len(response)} records instead of the expected {len(identifiers_batch)}')
                log.error(ex)
                raise ex
        except:
            log.info('will submit batch record per record')
            response = []
            for identifier in identifiers_batch:
                response.append(requests.post(ccd, headers=headers, data=identifier).json())
            dsstox_data.append(pd.DataFrame.from_records(chain.from_iterable(response)).assign(identifier=identifiers_batch))
    dsstox_data = pd.concat(dsstox_data, axis='index', ignore_index=True, sort=False)
    return dsstox_data

