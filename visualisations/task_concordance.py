import pandas as pd

import logger
log = logger.get_logger(__name__)

from pathlib import Path
import json

import numpy as np
import re

import seaborn as sns
import matplotlib.pyplot as plt

from cheminformatics.rdkit_toolkit import read_sdf
from sklearn.metrics import jaccard_score


def rename_task(name: str) -> str:
    '''
    Shortens the task name to create the heatmap. This may require additional patterns in case of exceptions.
    :param name: long task name
    :return: shorted task name
    '''

    # bacterial reverse mutation assay, Salmonella typhimurium, strain specified, metabolic activation specified
    p = re.compile(r'Salmonella typhimurium \(TA (\d+)\), (no|yes)')
    if m := p.search(name):
        strain = m.group(1)
        metabolic_activation = m.group(2)
        shortened_name = 'TA' + strain + ' ' + (
            'S9+' if metabolic_activation == 'yes' else 'S9-' if metabolic_activation == 'no' else metabolic_activation)
        return shortened_name

    # bacterial reverse mutation assay, Salmonella typhimurium, strain specified
    p = re.compile(r'Salmonella typhimurium \(TA (\d+)\)$')
    if m := p.search(name):
        strain = m.group(1)
        shortened_name = 'TA' + strain
        return shortened_name

    # bacterial reverse mutation assay, Escherichia coli, strain specified, metabolic activation specified
    p = re.compile(r'Escherichia coli \((.*)\), (no|yes)')
    if m := p.search(name):
        strain = m.group(1)
        metabolic_activation = m.group(2)
        shortened_name = 'E. coli ' + strain + ' ' + (
            'S9+' if metabolic_activation == 'yes' else 'S9-' if metabolic_activation == 'no' else metabolic_activation)
        return shortened_name

    # bacterial reverse mutation assay
    p = re.compile(r'bacterial reverse mutation assay$')
    if m := p.search(name):
        shortened_name = 'AMES'
        return shortened_name

    # in vitro mammalian chromosome aberration test
    p = re.compile(r'in vitro mammalian chromosome aberration test$')
    if m := p.search(name):
        shortened_name = 'mamm. chrom. abb.'
        return shortened_name

    # in vitro mammalian cell micronucleus test
    p = re.compile(r'in vitro mammalian cell micronucleus test$')
    if m := p.search(name):
        shortened_name = 'mamm. micronucleus'
        return shortened_name

    # in vitro gene mutation study in mammalian cells
    p = re.compile(r'in vitro gene mutation study in mammalian cells$')
    if m := p.search(name):
        shortened_name = 'mamm. gene mut.'
        return shortened_name

    # in vitro mammalian cell gene mutation test using the Hprt and xprt genes
    p = re.compile(r'in vitro mammalian cell gene mutation test using the Hprt and xprt genes$')
    if m := p.search(name):
        shortened_name = 'mamm. gene mut. HPRT/XPRT'
        return shortened_name

    # in vitro mammalian cell gene mutation test using the thymidine kinase genes
    p = re.compile(r'in vitro mammalian cell gene mutation test using the thymidine kinase gene$')
    if m := p.search(name):
        shortened_name = 'mamm. gene mutat. TK'
        return shortened_name


    log.info(f'cannot shorten task name {name}')
    return name


def visualise_task_concordance(outp_sdf: Path, outps: list[Path]) -> None:
    '''
    Creates a heatmap visualisation of the task concordance in terms of Jaccard similarity
    :param outp_sdf: path to the dataset used for modelling
    :param outps: paths to store the figure (concordance and co-occurrence)
    :return: None
    '''

    plt.interactive('off')

    mols = read_sdf(outp_sdf)
    all_data = []
    for i_mol, mol in enumerate(mols):
        mol_data = pd.DataFrame(json.loads(mol.GetProp('genotoxicity')))
        mol_data['i_mol'] = i_mol
        all_data.append(mol_data)
    all_data = pd.concat(all_data, axis='index', ignore_index=True, sort=False)
    # keep only positive and negative outcomes
    msk = all_data['genotoxicity'].isin(['positive', 'negative'])
    all_data = all_data.loc[msk]
    all_data['genotoxicity'] = all_data['genotoxicity'].map({'positive': 1, 'negative': 0})
    task_pair_data = []
    for task1 in all_data['task'].unique():
        for task2 in all_data['task'].unique():
            msk1 = all_data['task'] == task1
            msk2 = all_data['task'] == task2
            msk = msk1 | msk2
            tmp = all_data.loc[msk]
            tmp = tmp.pivot(index='i_mol', columns='task', values='genotoxicity')
            tmp = tmp.dropna(how='any')
            jaccard = jaccard_score(tmp[task1], tmp[task2])
            task_pair_data.append({'task 1': task1, 'task 2': task2, 'jaccard similarity': jaccard, 'support': len(tmp)})
    # task concordance
    task_correlations = pd.DataFrame(task_pair_data)
    task_correlations[['task 1', 'task 2']] = task_correlations[['task 1', 'task 2']].map(rename_task).dropna()
    task_correlations = task_correlations.pivot(index='task 1', columns='task 2', values='jaccard similarity')
    task_correlations = task_correlations.sort_index(axis='index', ascending=False).sort_index(axis='columns', ascending=False)
    # support
    support = pd.DataFrame(task_pair_data)
    support[['task 1', 'task 2']] = support[['task 1', 'task 2']].map(rename_task).dropna()
    support = support.pivot(index='task 1', columns='task 2', values='support')
    support = support.sort_index(axis='index', ascending=False).sort_index(axis='columns', ascending=False)
    support = support.fillna(0)

    # task concordance
    # .. create a mask for the upper triangle
    mask = np.triu(np.ones_like(task_correlations, dtype=bool))
    task_correlations = task_correlations.iloc[1:, :-1]
    mask = mask[1:, :-1]

    # Set up the matplotlib figure for he test concordance
    fig = plt.figure(figsize=(8, 6))
    ax = fig.subplots()
    sns.heatmap(task_correlations, mask=mask, annot=True, fmt='.2f', cmap='Blues', square=True, vmin=0, vmax=1, ax=ax, annot_kws={"fontsize": 8} )
    ax.set_xlabel(None, fontsize=8)
    ax.set_ylabel(None, fontsize=8)

    # change the font size of the tick pabels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

    # change the fontsize of the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(outps[0], dpi=600)
    plt.close(fig)


    # data availability and task co-occurrence
    mask = np.triu(np.ones_like(support, dtype=bool), k=1)

    # Set up the matplotlib figure for he test concordance
    fig = plt.figure(figsize=(8, 6))
    ax = fig.subplots()
    sns.heatmap(support, annot=True, mask=mask, fmt='d', cmap='Blues', square=True, ax=ax, annot_kws={"fontsize": 6} )
    ax.set_xlabel(None, fontsize=8)
    ax.set_ylabel(None, fontsize=8)

    # change the font size of the tick pabels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

    # change the fontsize of the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(outps[1], dpi=600)
    plt.close(fig)


    plt.interactive('on')
