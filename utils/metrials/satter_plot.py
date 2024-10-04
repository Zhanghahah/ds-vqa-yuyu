"""
some scatter plot for SI

"""

"1. single-objective optimization using Chemma"
import pandas as pd
import os
from rdkit import Chem
import numpy as np
import pickle
def canonical_smiles(smi):
    """
        Canonicalize a SMILES without atom mapping
        """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    else:
        canonical_smi = Chem.MolToSmiles(mol)
        # print('>>', canonical_smi)
        if '.' in canonical_smi:
            canonical_smi_list = canonical_smi.split('.')
            canonical_smi_list = sorted(
                canonical_smi_list, key=lambda x: (len(x), x)
            )
            canonical_smi = '.'.join(canonical_smi_list)
        return canonical_smi
def load_data(hte_data_path):
    data_df = pd.read_csv(hte_data_path)
    data_df['catalyst_preds'] = data_df['catalyst_preds'].apply(lambda x: eval(x))
    data_df['solvents'] = data_df['solvents'].apply(lambda x: '.'.join(list(set(map(str.strip, sorted(x.split('.')))))) if x==x else x)
    data_df['catalysts'] = data_df['catalysts'].apply(lambda x: '.'.join(list(set(map(str.strip, sorted(x.split('.')))))) if x==x else x)
    data_df['catalyst_preds'] = data_df['catalyst_preds'].apply(lambda list_x: ['.'.join(list(set(map(str.strip,
                                                                                                      sorted(x.split('.'))
                                                                                                      )))
                                                                                         )
                                                                 if x==x else x for x in list_x])
    if "reactants" in data_df.columns:
        data_df['reactants'] = data_df['reactants'].apply(lambda x: eval(x))
        data_df['products'] = data_df['products'].apply(lambda x: eval(x))
    return data_df
def common_condition_pred_vis(hte_data_path, raw_data_path):
    data_df = load_data(hte_data_path)
    raw_df = pd.read_csv(raw_data_path)
    group_df = data_df.groupby(['reagents', 'solvents'])
    raw_group_df = raw_df.groupby(['reagents', 'solvents', 'catalysts'], as_index=False).agg({'yield': 'max'})
    raw_group_df['catalysts'] = raw_group_df['catalysts'].apply(lambda x: sorted(x.split("."), key=lambda y: len(y))[-1])
    raw_group_df['reagents'] = raw_group_df['reagents'].apply(lambda x: canonical_smiles(x))
    raw_group_df['solvents'] = raw_group_df['solvents'].apply(lambda x: canonical_smiles(x))
    pred_dict = {}
    raw_dict = {}
    # for group_name, df_g in raw_group_df:
    #     for row_index, raw_row in df_g.iterrows():
    #         ligand = canonical_smiles(raw_row['catalysts']).split('.')[-1]
    #         c_unique_key = list(group_name)
    #         c_unique_key.append(ligand)
    #         raw_dict[tuple(c_unique_key)] = raw_row['yield']

    for group_name, df_g in group_df:
        for row_index, row in df_g.iterrows():
            pred_catalysts = [canonical_smiles(item) for item in row['catalyst_preds']]

            for pred_idx, pred_cat in enumerate(pred_catalysts):
                # need to update
                rxn_key = list(group_name)
                pred_ligand = sorted(pred_cat.split("."), key=lambda x: len(x))[-1]
                rxn_ligand_df = raw_group_df[(raw_group_df['reagents'] == rxn_key[0]) & (raw_group_df['solvents'] == rxn_key[1])]
                try:
                    observed_yield = rxn_ligand_df[rxn_ligand_df['catalysts'] == pred_ligand]['yield']
                    rxn_key.append(pred_ligand)
                    pred_dict[tuple(rxn_key)] = (observed_yield, pred_idx)
                except:
                    continue
    return pred_dict

def cn_process_pred_vis(hte_data_path, raw_data_path):
    pred_dict = {}
    raw_df = pd.read_csv(raw_data_path)
    raw_df['base_smiles'] = raw_df['base_smiles'].apply(lambda x: canonical_smiles(x))
    raw_df['substrate_smiles'] = raw_df['substrate_smiles'].apply(lambda x: canonical_smiles(x))
    raw_df['additive_smiles'] = raw_df['additive_smiles'].apply(lambda x: canonical_smiles(x))


    data_df = pd.read_csv(hte_data_path)
    data_df['catalyst_preds'] = data_df['catalyst_preds'].apply(lambda x: eval(x))
    group_df = data_df.groupby(['base_smiles', 'substrate_smiles', 'additive_smiles'])
    for group_name, df_g in group_df:
        for row_index, row in df_g.iterrows():
            pred_catalysts = [canonical_smiles(item) for item in row['catalyst_preds']]
            for pred_idx, pred_cat in enumerate(pred_catalysts):
                # need to update
                rxn_key = list(group_name)
                pred_ligand = sorted(pred_cat.split("."), key=lambda x: len(x))[-1]
                rxn_ligand_df = raw_df[(raw_df['base_smiles'] == rxn_key[0])
                                       & (raw_df['substrate_smiles'] == rxn_key[1])
                                       & (raw_df['additive_smiles'] == rxn_key[2])]
                try:
                    observed_yield = rxn_ligand_df[rxn_ligand_df['ligand_smiles'] == pred_ligand]['yield']
                    rxn_key.append(pred_ligand)
                    pred_dict[tuple(rxn_key)] = (observed_yield, pred_idx)
                except:
                    continue

def BH_condition_pred_vis(hte_data_path):
    pred_dict = {}
    data_df = pd.read_csv(hte_data_path)
    data_cols = data_df.columns.tolist()
    # reactants,products,catalysts,base,solvents,yield,catalyst_pred
    for col_name in ["reactants", "products", "catalyst_preds"]:
        data_df[col_name] = data_df[col_name].apply(lambda x: eval(x))

    for col_name in ["reactants", "products"]:
        data_df[col_name] = data_df[col_name].apply(lambda x: '.'.join(x))

    for col_name in ["catalysts", "base", "solvents"]:
        data_df[col_name] = data_df[col_name].apply(lambda x: canonical_smiles(x))

    raw_group_df = data_df.groupby(['reactants', 'products', 'catalysts', 'base', 'solvents'], as_index=False).agg({'yield': 'max'})

    group_df = data_df.groupby(['reactants', 'products', 'base', 'solvents'])

    for group_name, df_g in group_df:
        for row_index, row in df_g.iterrows():
            pred_catalysts = [canonical_smiles(item) for item in row['catalyst_preds']]
            for pred_idx, pred_cat in enumerate(pred_catalysts):
                # need to update
                rxn_key = list(group_name)
                pred_ligand = sorted(pred_cat.split("."), key=lambda x: len(x))[-1]
                rxn_ligand_df = raw_group_df[(raw_group_df['reactants'] == rxn_key[0])
                                       & (raw_group_df['products'] == rxn_key[1])
                                       & (raw_group_df['base'] == rxn_key[2])
                                        & (raw_group_df['solvents'] == rxn_key[3])]

                observed_yield = rxn_ligand_df[rxn_ligand_df['catalysts'] == pred_ligand]['yield']
                if len(observed_yield.values) > 0:
                    rxn_key.append(pred_ligand)
                    pred_dict[tuple(rxn_key)] = (observed_yield.values[0], pred_idx)

    return pred_dict

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


### B: /data/zhangyu/yuruijie/yuyu_paper_plot/scatter_0208/metrials/top_0_seq3_src_gt_preds_condition_optim_plot.pkl
### C: /data/zhangyu/yuruijie/yuyu_paper_plot/scatter_0208/metrials/11-07_BH_seq3_src_gt_preds_condition_optim_plot.pkl

if __name__ == '__main__':
    hte_data_path_prefix = "/data/zhangyu/process_vis_data"
    vis_data_b = "/data/zhangyu/yuruijie/yuyu_paper_plot/scatter_0208/metrials/top_0_seq3_src_gt_preds_condition_optim_plot.pkl"
    vis_data_c = "/data/zhangyu/qihang/code/11-07_BH_seq3_src_gt_preds_condition_optim_plot_78.pkl"
    vis_data_dict = load_pkl(vis_data_c)
    # top 0:
    # cn_process:
    data_tag = 'BH'
    dataset_file = "11-07_BH_seq3_src_gt_preds.csv"
    data_name = dataset_file.split(".")[0]
    hte_data_path = os.path.join(hte_data_path_prefix, dataset_file)
    raw_data_path = os.path.join(hte_data_path_prefix, "cn-processed.csv")
    if data_tag == 'cn':
        pred_dict = cn_process_pred_vis(hte_data_path, raw_data_path)
    elif data_tag == 'top':
        pred_dict = common_condition_pred_vis(hte_data_path, raw_data_path)
    elif data_tag == 'BH':
        pred_dict = BH_condition_pred_vis(hte_data_path)
        with open(f'./{data_name}_condition_optim.pkl', 'wb') as f:
            pickle.dump(pred_dict, f)

    # np.save(f'./{data_name}_condition_optim.npy', pred_dict)

