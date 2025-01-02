import logger
log = logger.get_logger(__name__)

import pandas as pd
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

def visualise_database_concordance(outp_tab: Path, output_path: Path) -> None:
    '''
    Visualises the concordance among the databases by creating one figure per task
    :param outp_tab: path to the combined dataset
    :param output_path: path to folder where the figures will be saved,
    '''

    # prepare the data
    data = pd.read_excel(outp_tab)
    data['genotoxicity details'] = data['genotoxicity details'].apply(lambda x: json.loads(x))
    data = data.rename({'genotoxicity': 'genotoxicity (overall)'}, axis='columns')
    data = data.explode('genotoxicity details').reset_index(drop=True)
    data = pd.concat([data[['smiles_std', 'task', 'genotoxicity (overall)']], pd.DataFrame.from_records(data['genotoxicity details'].to_list())], axis='columns', ignore_index=False, sort=False)
    sources = data['source'].drop_duplicates().to_list()
    data = data.pivot(index=['task', 'smiles_std'], columns='source', values='genotoxicity (source)').fillna('not available').reset_index()
    # genotoxicity is understood as conflicting if there is both a positive and a negative genotoxicity call
    data['conflicting genotoxicity'] = data[sources].apply(lambda x: 'yes' if len({'positive', 'negative'}.intersection(set(x)))==2 else 'no', axis='columns')
    data.to_excel(output_path/'database_concordance.xlsx')
    data[sources] = data[sources].replace({'positive': '+', 'negative': '-', 'ambiguous': 'A', 'not available': None})
    data_summary = data.groupby(['task', 'conflicting genotoxicity']+sources, dropna=False)['smiles_std'].nunique().rename('number of structures').reset_index()
    data_summary = data_summary.sort_values(by=['task', 'number of structures'], ascending=[True, False])


    plt.interactive(False)
    # loop over the tasks and visualise the database concordance for each task separately
    tasks = data_summary['task'].drop_duplicates().to_list()
    for task in tasks:
        log.info(f'visualising database concordance for task: {task}')

        # prepare the dataframe
        tmp = data_summary.loc[data_summary['task']==task][sources+['conflicting genotoxicity', 'number of structures']]
        # set the number of bars (roughly) to see for each "conflicting genotoxicity" value
        n_bars = 15
        try:
            min_number_structures_conflicting = tmp.loc[tmp['conflicting genotoxicity']=='yes', 'number of structures'].sort_values(ascending=False).iloc[n_bars]
        except IndexError:
            min_number_structures_conflicting = 1
        try:
            min_number_structures_not_conflicting = tmp.loc[tmp['conflicting genotoxicity']=='no', 'number of structures'].sort_values(ascending=False).iloc[n_bars]
        except:
            min_number_structures_not_conflicting = 1
        msk = ((tmp['conflicting genotoxicity']=='yes') & (tmp['number of structures']>=min_number_structures_conflicting)
               | (tmp['conflicting genotoxicity']=='no') & (tmp['number of structures']>=min_number_structures_not_conflicting)
               )
        tmp = tmp.loc[msk]
        tmp = tmp.T
        tmp = tmp.dropna(axis='index', how='all')
        tmp = tmp.fillna('')

        # create figure and subplots
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, gridspec_kw={'height_ratios': [3, 1, 0.1]}, figsize=(10, 5))

        # plot the bar plot in top subplot
        bars = ax1.bar(range(tmp.shape[1]), tmp.loc['number of structures'])
        ax1.set_ylabel('number of structures')
        ax1.set_yscale('log')
        ax1.set_xlim([-0.5, tmp.shape[1]-0.5])
        # .. set the spines and remove the horizontal ticks
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_position(('outward', 10))
        plt.setp(ax1, 'xticks', [])
        # .. add text labels above the bars, and the set the colour
        for i_col, bar in enumerate(bars):
            # set the text label
            height = bar.get_height()*1.1
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f'{bar.get_height():d}', ha='center', va='bottom', fontsize=7, rotation=90)
            # set the colour
            is_conflicting = tmp.iloc[:, i_col]['conflicting genotoxicity'] == 'yes'
            all_positive = (tmp.iloc[:, i_col].isin(['+']).sum()>0) & (tmp.iloc[:, i_col].isin(['-', 'A']).sum()==0)
            all_negative = (tmp.iloc[:, i_col].isin(['-']).sum()>0) & (tmp.iloc[:, i_col].isin(['+', 'A']).sum()==0)
            all_ambiguous = (tmp.iloc[:, i_col].isin(['A']).sum()>0) & (tmp.iloc[:, i_col].isin(['+', '-']).sum()==0)
            if all_positive:
                # all positive
                bar.set_color('#e28743')
            elif all_ambiguous:
                # all ambiguous
                bar.set_color('w')
                bar.set_edgecolor('k')
            elif all_negative:
                # all negative
                bar.set_color('#1e81b0')
            else:
                # conflicting
                bar.set_color('#babbbb')

        # create table with genotoxicity outcomes in the middle subplot
        table_data = tmp.drop(['number of structures', 'conflicting genotoxicity'], axis='index')
        table = ax2.table(cellText=table_data.values, colLabels=None, cellLoc='center', loc='center', edges='closed', rowLabels=table_data.index.to_list())
        # .. set the grid cell lines thinner
        for key, cell in table.get_celld().items():
            cell.set_fontsize(8)
            cell.set_linewidth(0.5)
            if key[1] == -1:  # row labels
                cell.set_edgecolor('none')
                cell.get_text().set_ha('left')
        # .. hide the axes of the table subplot
        ax2.axis('off')

        # create table with number of databases in the bottom subplot
        table_data = tmp.drop(['number of structures', 'conflicting genotoxicity'], axis='index').isin(['+', '-', 'A']).sum(axis='index').rename('number of sources').to_frame().T
        table = ax3.table(cellText=table_data.values, colLabels=None, cellLoc='center', loc='center', edges='closed', rowLabels=table_data.index.to_list())
        # .. set the font size
        for key, cell in table.get_celld().items():
            cell.set_fontsize(8)
            cell.set_linewidth(0.5)
            if key[1] == -1:  # row labels
                cell.set_edgecolor('none')
                cell.get_text().set_ha('left')
        # .. hide the axes of the table subplot
        ax3.axis('off')

        # adjust layout to ensure alignment and save figure
        plt.tight_layout()
        fig.savefig(output_path/f'database_concordance_{task}.png', dpi=600)
        plt.close(fig)

    plt.interactive(True)

