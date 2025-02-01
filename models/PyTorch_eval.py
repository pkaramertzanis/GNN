import numpy as np

import logger
log = logger.get_logger(__name__)

import pandas as pd

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


from rdkit import Chem
from tqdm import tqdm
import torch.nn.functional as F

from rdkit.Chem import AllChem

# set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval(mols: list[Chem.Mol],
         net,
         tasks: list[str],
         fingerprint_parameters: dict,) -> pd.DataFrame:
    '''
    Generate predictions for a list of molecules using a trained model. This function does not examine if
    the molecules are suitable for running the model. It also does not standardise the molecular structures. These
    operations need to be performed before calling this function.
    :param mols: list of RDKit molecules to run predictions or
    :param net: trained PyTorch model
    :param tasks: list with task names that the model was trained to predict
    :param fingerprint_parameters: dictionary with fingerprint paramters
    :return: pandas dataframe with the predictions
    '''

    # set the fingerprint generator
    fpgen = AllChem.GetMorganGenerator(radius=fingerprint_parameters['radius'], fpSize=fingerprint_parameters['fpSize'],
                                       countSimulation=False, includeChirality=False)

    # compute the fingerprints
    fgs = []
    for i_mol, mol in tqdm(enumerate(mols)):
        fg = fpgen.GetFingerprint(mol)
        fg_count = fpgen.GetCountFingerprint(mol)
        if fingerprint_parameters['type'] == 'binary':
           fg = ''.join([str(bit) for bit in fg.ToList()])  # fg.ToBitString()
        elif fingerprint_parameters['type'] == 'count':
            fg = ''.join([str(min(bit, 9)) for bit in fg_count.ToList()])
        else:
            ex = ValueError(f"unknown fingerprint type: {fingerprint_parameters}")
            log.error(ex)
            raise ex
        fgs.append(fg)
    X = np.array([[float(bit) for bit in list(fg)] for fg in fgs], dtype=np.float32)

    net.eval()

    # create the dataset and the data loader
    dset = TensorDataset(torch.tensor(X))
    batch_size = 1024
    # .. we eneed to ensure that we do not suffle here
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=False, drop_last=False)

    predictions = []
    i_mol_start = 0
    for i_batch, batch in enumerate(dataloader):
        for i_task, task in enumerate(tasks):
            batch[0] = batch[0].to(device)
            with torch.no_grad():
                pred = net(batch[0], task_id=i_task)
            pred = pd.DataFrame(F.softmax(pred, dim=1).detach().cpu().numpy(),
                                columns=['negative (probability)', 'positive (probability)'])
            pred['genotoxicity call'] = np.where(pred['positive (probability)'] >= 0.5, 'positive', 'negative')
            pred['task'] = task
            pred['i mol'] = range(i_mol_start, i_mol_start + len(batch[0]))
            predictions.append(pred)
        i_mol_start = i_mol_start + len(batch[0])
    predictions = pd.concat(predictions, axis='index', ignore_index=True, sort=False)

    return predictions