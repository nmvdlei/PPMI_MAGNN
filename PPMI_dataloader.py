import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
from pathlib import Path
import warnings
import data_utils as du
data_dir = du.find_data_dir('app')
warnings.filterwarnings("ignore")

#######################################################################################################################################################################

# Heterogeneous graph representation learning for protein-protein-metabolite interaction networks

# This script contains 3 classes: 
#   1. DataLoader
#   2. ExerciseMetabolomicsDataLoader
#   3. MetaboliteNameMatcher

#######################################################################################################################################################################

class DataLoader():
    """ This class preprocesses all data required to apply HGRL to a PPMI network. The calculations happen at initalization.
        Finally, the data of interest is the 4 data components:
          1. PPMI network: self.PPMI_pruned
          2. Metabolite attributes: self.X
          3. Protein attributes: self.protein_features
          4. Metabolite class labels: self.y
    """
    def __init__(self, publication_names, include_feature_category, metabolite_matcher_file=None, calc_CCS_settings=None, allowed_ccs_values=[-1, 0, 1], min_metabolite_per_feature=6, min_proteins_per_feature=3):
        #Store preprocessing settings 
        self.include_feature_category = include_feature_category
        self.allowed_ccs_values = allowed_ccs_values
        self.publication_names = publication_names
        
        #Obtain CCS data for specific publication(s)
        publications = []
        for publication_name in self.publication_names:
            publications.append(ExerciseMetabolomicsDataLoader(publication_name, calc_CCS_settings))
            
        self.millan_2020 = publications[0]
        
        #Load or create metabolite matcher file to obtain HMDB IDs for each datapoint        
        #The metabolite matcher takes a list of metabolite names as input and tries to find HMDB and KEGG ids for each string 
        if metabolite_matcher_file is not None:
            self.metabolite_matcher = du.read_from_pickle(metabolite_matcher_file)
        else:
            self.metabolite_matcher = MetaboliteNameMatcher(list(self.millan_2020.log2fold_change.index))
        
        #hmdb_log2fold_change_CSS is a pandas DataFrame with 2 columns. The row index is HMDB accesion IDs. The 2 columns are mean log2fold change and CCS 
        self.hmdb_log2fold_change_CSS = self.get_hmdb_metabolites_with_data()
        
        #It may be the case (due to metabolite name matching) that duplicate HMDB IDs are identified. 
        #Since it is hard to make a decision on which version of duplicate IDs' data to trust, all duplicates are removed
        self.hmdb_log2fold_change_CSS = self.get_non_duplicate_rows(self.hmdb_log2fold_change_CSS)
        
        #Uses may decide to use only [-1, 1] or all [-1, 0, 1] to make the classification problem setting binary or multi class
        self.hmdb_log2fold_change_CSS = self.get_rows_with_allowed_values(self.hmdb_log2fold_change_CSS)

        #The PPMI_full is a networkx graph and may be constructed from a pandas DataFrame with edges
        #Edge weights are still included in this step of the process, but are omitted later on.
        self.PPMI_full = self.convert_interactome_to_graph(self.get_interactome())
        self.remove_isolated_nodes_from_full_PPMI()
        
        #Only protein nodes with attribute information should be included
        #protein_nodes_gene is a pandas Series which lists the Gene Symbol / Entrez ID for each node if it was found
        #Some 583 nodes from the PPMI were removed due to this criterium
        self.protein_nodes_gene = self.match_protein_nodes_to_gene()
        self.remove_unmatched_proteins_from_full_PPMI()
        
        #Only metabolites for which CCS data is available and that exist in the full PPMI should be included in the pruned PPMI
        #The pruning process is described in preprocess_PPMI.ipynb and the thesis manuscript
        #The main goal of the pruning process is to have target class labels for all metabolite nodes in the network and to reduce network size as much as possible for reducing computational cost
        self.hmdb_log2fold_change_CSS = self.get_metabolites_in_PPMI(self.hmdb_log2fold_change_CSS)
        self.PPMI_pruned = self.prune_PPMI_network(self.hmdb_log2fold_change_CSS.index)
        self.remove_isolated_nodes_from_pruned_PPMI()
        
        #With the PPMI network pruned and the set of metabolites and proteins defined, the target class labels and attribute matrices can be constructed
        #y is the target class label. This work could be extended to allow for regression and the mean log2fold change could be used 
        #The self.include_feature_category are used for this
        self.y = self.hmdb_log2fold_change_CSS['CCS']
        self.X = self.construct_metabolite_feature_df(min_metabolite_per_feature)
        self.protein_features = self.construct_protein_feature_df(min_proteins_per_feature)
        
        #Since NaN values in attribute matrices cause problems in the HGRL algorithms and classifiers, these columns are removed. 
        #Mostly the physical properties features of metabolites are affected by this, as these values are unavailable from some/most metabolites 
        self.drop_feature_columns_with_NaNs()

    def get_interactome(self):
        prot_hmdb_interactome_file = du.get_file_path(data_dir, 'interactome', 'txt', 'prot_hmdb_interactome.txt')
        prot_hmdb_interactome = pd.read_csv(prot_hmdb_interactome_file, sep="\t")
        return prot_hmdb_interactome

    def convert_interactome_to_graph(self, interactome):
        G = nx.Graph()
        nodes = np.unique(np.concatenate((interactome['protein1'].unique(), interactome['protein2'].unique())))
        G.add_nodes_from(nodes)
        edges = interactome[['protein1', 'protein2', 'cost']].values.tolist()
        G.add_weighted_edges_from(edges, weight='cost')
        return G        
        
    def get_hmdb_metabolites_with_data(self):
        hmdb_id = self.metabolite_matcher.name_accessions.set_index('name')['hmdb_accession']
        CCS = self.millan_2020.CCS
        log2foldchange = self.millan_2020.log2fold_change

        hmdb_log2fold_change_CSS = pd.DataFrame([hmdb_id, log2foldchange, CCS], index=['hmdb_accession', 'log2foldchange', 'CCS']).T.set_index('hmdb_accession')
        hmdb_log2fold_change_CSS = hmdb_log2fold_change_CSS[hmdb_log2fold_change_CSS.index.notna()]
        return hmdb_log2fold_change_CSS
    
    def get_non_duplicate_rows(self, hmdb_log2fold_change_CSS):
        self.duplicated_ids = hmdb_log2fold_change_CSS[hmdb_log2fold_change_CSS.index.duplicated()].index
        self.duplicated_ids_list = list(self.duplicated_ids)
        non_duplicated = self.hmdb_log2fold_change_CSS.drop(self.duplicated_ids)
        return non_duplicated

    def get_rows_with_allowed_values(self, hmdb_log2fold_change_CSS, in_out='in'):
        if in_out == 'in':
            use = [CCS in self.allowed_ccs_values for CCS in hmdb_log2fold_change_CSS['CCS']]
        elif in_out == 'out':
            use = [CCS not in self.allowed_ccs_values for CCS in hmdb_log2fold_change_CSS['CCS']]
            
        rows_with_allowed_values = hmdb_log2fold_change_CSS.loc[use]
        return rows_with_allowed_values

    def get_metabolites_in_PPMI(self, hmdb_log2fold_change_CSS, in_out='in'):
        hmdb_ids_in_both = self.match_exerc_metab_to_nodes(hmdb_log2fold_change_CSS, in_out)
        return hmdb_log2fold_change_CSS.loc[hmdb_ids_in_both]
    
    def match_exerc_metab_to_nodes(self, hmdb_log2fold_change_CSS, in_out):
        metabolites_in_PPMI_full = self.get_metabolite_nodes()
        metabolites_in_exerc_metab = hmdb_log2fold_change_CSS.index
        
        df = pd.DataFrame([(metabolite, self.in_list(metabolite, metabolites_in_PPMI_full) ) for metabolite in metabolites_in_exerc_metab], columns=['hmdb_accession', 'in_interactome'])

        if in_out=='in':
            in_interactome = df[df['in_interactome']]
            return list(in_interactome['hmdb_accession'])
        elif in_out=='out':
            not_in_interactome = df[~df['in_interactome']]
            return list(not_in_interactome['hmdb_accession'])

    def get_gene_from_synonym(self, synonym, gene_synonym_dict):
        try:
            return gene_synonym_dict[synonym]
        except KeyError:
            return None
        
    def remove_unmatched_proteins_from_full_PPMI(self):        
        node_gene = self.protein_nodes_gene
        unmatched_protein_nodes = list(node_gene[node_gene.isna()].index)
        
        for node in unmatched_protein_nodes:
            if node in self.PPMI_full.nodes:
                self.PPMI_full.remove_node(node)        
        
    def match_protein_nodes_to_gene(self):
        gene_synonym_dict_file = du.get_file_path(data_dir, 'ProteinAtlas proteins', 'protein matching', 'gene_synonym_dict.p')
        gene_synonym_dict = du.read_from_pickle(gene_synonym_dict_file)

        protein_class_file = du.get_file_path(data_dir, 'ProteinAtlas proteins', 'Feature dfs pickle', 'protein_class.p')
        protein_class_columns = du.read_from_pickle(protein_class_file)
        
        protein_nodes = pd.Series([node for node in list(self.PPMI_full.nodes) if not 'HMDB' in node])        
        protein_node_in_index = pd.Series([protein_node in protein_class_columns.index for protein_node in protein_nodes])

        protein_nodes_with_features = protein_nodes[protein_node_in_index]
        protein_nodes_without_features = protein_nodes[~protein_node_in_index]
        protein_matching_using_dict = pd.Series([self.get_gene_from_synonym(node, gene_synonym_dict) for node in protein_nodes_without_features], index=protein_nodes_without_features.index)

        df_result = pd.DataFrame(protein_nodes, columns=['Node'])
        df_result['Gene'] = [None]*len(protein_nodes)
        df_result.loc[protein_nodes_with_features.index, 'Gene'] = protein_nodes_with_features
        df_result.loc[protein_matching_using_dict.index, 'Gene'] = protein_matching_using_dict
        df_result = df_result.set_index('Node')

        return df_result['Gene']

    def print_protein_names(self, list_of_proteins):
        print("'"+"', '".join(list_of_proteins)+"'")
        
    def get_metabolite_nodes(self):
        all_nodes = list(self.PPMI_full.nodes)
        metabolites_in_PPMI = [m for m in all_nodes if 'HMDB' in m]
        return metabolites_in_PPMI

    def get_protein_nodes(self):
        all_nodes = list(self.PPMI_pruned.nodes)
        proteins_in_PPMI = [m for m in all_nodes if 'HMDB' not in m]
        return proteins_in_PPMI
    
    def in_list(self, value, listing):
        return value in listing

    def target_distribution_stats(self):
        df = pd.DataFrame([self.y.value_counts(), self.y.value_counts()/len(self.y)*100], index=['Amount', 'Percentage']).T
        total = pd.DataFrame({"Amount": len(self.y),
                 "Percentage": 100.}, index=["Total"])
        df = df.append(total)
        return df.style.format({'Amount': "{:.0f}", "Percentage": "{:.1f}%"})
    
    def print_settings(self):
        print('calc_CCS_settings')
        for k, v in self.millan_2020.calc_CCS_settings.items():
            print(f"   {k}: '{v}'")
        print('allowed CCS values')
        print(f'   {self.allowed_ccs_values}')
        
        print('included feature categories')
        for k, v in self.include_feature_category.items():
            print(f"   {k}: '{v}'")
        print('')

    def drop_feature_columns_with_NaNs(self):
        columns_with_NaNs = self.X.isna().sum()[self.X.isna().sum() > 0].index.to_list()
        self.X = self.X.drop(columns_with_NaNs, axis=1)
        
    def construct_metabolite_feature_df(self, min_metabolite_per_feature=6):
        metabolite_features_dfs_list = [] 

        for feature_name, include in self.include_feature_category['metabolite'].items():
            if include:
                file_name = du.get_file_path(data_dir, 'HMDB metabolites', 'Feature dfs pickle', f'hmdb_metabolites_{feature_name}.p')
                feature_df = du.read_from_pickle(file_name)
                metabolite_features_dfs_list.append(feature_df)

        metabolite_features_all_df = pd.concat(metabolite_features_dfs_list, axis=1)        
        metabolite_features_df = metabolite_features_all_df.loc[self.hmdb_log2fold_change_CSS.index]
        
        #Remove those columns with only a single unique value
        columns_with_single_unique_value = metabolite_features_df.columns[metabolite_features_df.nunique()==1]
        metabolite_features_df = metabolite_features_df.drop(columns_with_single_unique_value, axis=1)
        
        correct_feature_column = np.array(pd.DataFrame(metabolite_features_df == 0).sum() >= min_metabolite_per_feature) & np.array(pd.DataFrame(metabolite_features_df == 1).sum() >= min_metabolite_per_feature)
        float_columns = np.array(metabolite_features_df.dtypes == np.dtype('float64'))
        columns_to_include = correct_feature_column | float_columns #correct feature column or float column 
        
        metabolite_features_df = metabolite_features_df.loc[:,columns_to_include]
        
        return metabolite_features_df
    
    def construct_protein_feature_df(self, min_proteins_per_feature=3):
        protein_features_dfs_list = [] 

        for feature_name, include in self.include_feature_category['protein'].items():
            if include:
                file_name = du.get_file_path(data_dir, 'ProteinAtlas proteins', 'Feature dfs pickle', f'{feature_name}.p')
                feature_df = du.read_from_pickle(file_name)
                protein_features_dfs_list.append(feature_df)

        protein_features_all_df = pd.concat(protein_features_dfs_list, axis=1)        
        protein_features_df = protein_features_all_df.loc[self.protein_nodes_gene[self.get_protein_nodes()]]
        
        #Remove those columns with only a single unique value
        columns_with_single_unique_value = protein_features_df.columns[protein_features_df.nunique()==1]
        protein_features_df = protein_features_df.drop(columns_with_single_unique_value, axis=1)

        protein_features_df = protein_features_df.loc[:,pd.DataFrame(protein_features_df == 0).sum() >= min_proteins_per_feature]
        protein_features_df = protein_features_df.loc[:,pd.DataFrame(protein_features_df == 1).sum() >= min_proteins_per_feature]
        
        return protein_features_df
    
    def prune_PPMI_network(self, nodes, level=2):
        G = self.PPMI_full
        G_small = nx.Graph()
        G_small.add_nodes_from(nodes)
        for metabolite in nodes:
            neighbors = list(G[metabolite])
            edges = list(zip([metabolite] * len(neighbors), neighbors))
            G_small.add_edges_from(edges) 

            if level == 2:
                G_small = self.add_neighbors(G, G_small, neighbors)
        return G_small
    
    def add_neighbors(self, G_old, G_new, neighbors):
        for neighbor in neighbors:
            neighbor_neighbors = list(G_old[neighbor])
            hmdb_neighbors = [node for node in neighbor_neighbors if 'HMDB' in node]
            prot_neighbors = [node for node in neighbor_neighbors if not 'HMDB' in node]

            neighbor_edges = list(zip([neighbor] * len(neighbor_neighbors), neighbor_neighbors))

            hmdb_neighbor_edges = list(zip([neighbor] * len(hmdb_neighbors), hmdb_neighbors))
            prot_neighbor_edges = list(zip([neighbor] * len(prot_neighbors), prot_neighbors))

            G_new.add_nodes_from(prot_neighbors)
            G_new.add_edges_from(prot_neighbor_edges)

        return G_new
    
    def remove_isolated_nodes_from_full_PPMI(self):
        #The specific nodes below are isolted nodes that are not connected to any of the other nodes in the full PPMI network and are removed
        nodes_to_remove = ['HMDB0000562', 'HMDB0001036', 'AOPEP', 'SLC47A1', 'SLC47A2', 'PRHOXNB']

        for node in nodes_to_remove:
            if node in self.PPMI_full.nodes:
                self.PPMI_full.remove_node(node)
    
    def remove_isolated_nodes_from_pruned_PPMI(self):
        #By pruning the PPMI network this pair of nodes becomes isolated and is therefore also removed
        nodes_to_remove = ['ALLC', 'HMDB0001209']

        for node in nodes_to_remove:
            if node in self.PPMI_pruned.nodes:
                self.PPMI_pruned.remove_node(node)
                if 'HMDB' in node:
                    self.hmdb_log2fold_change_CSS = self.hmdb_log2fold_change_CSS.drop(node)
                    
    def print_components(self, G):
        #To check the connectedness of nodes the connected components can be printed seperately. This can be done for the full or pruned PPMI
        i = 1
        for component in nx.connected_components(G):
            print(f'  - component {i}: {len(component)} nodes')
            if len(component) < 10:
                print(component)
            i+=1
    
    def print_data_counts(self):
        print('Number of metabolites in publication Millan 2020:', len(self.millan_2020.log2fold_change))
        print('Number of unnamed metabolites in publication Millan 2020:', len(self.metabolite_matcher.unnamed_metabolites))
        
        after_removing_unnamed = self.metabolite_matcher.name_accessions.set_index('name').drop(self.metabolite_matcher.unnamed_metabolites['name'])
        print('Number of metabolites in publication Millan 2020 after removing unnamed:', after_removing_unnamed.shape[0])
        
        with_kegg_no_hmdb = after_removing_unnamed[after_removing_unnamed['hmdb_accession'].isna()]
        print('Number of metabolites with KEGG id, without HMDB id:', with_kegg_no_hmdb.shape[0])
        
        print('Number of metabolites with HMDB id and data:', self.get_hmdb_metabolites_with_data().shape[0])
        
        print('Number of unique duplicated HMDB ids:', len(self.duplicated_ids.unique()))

        print('Number of removed rows becuase target value was not allowed:', self.get_rows_with_allowed_values(self.get_hmdb_metabolites_with_data(), in_out='out').shape[0])
        
        print('Total number of metabolites in full PPMI:', len(self.get_metabolite_nodes()))
        
        print('Total number of metabolites with CCS and not in PPMI:', self.get_metabolites_in_PPMI(self.get_hmdb_metabolites_with_data(), in_out='out').shape[0])
        
        print('Number of metabolites with CCS data in full PPMI:', self.get_metabolites_in_PPMI(self.get_hmdb_metabolites_with_data()).shape[0])

        print('')
        
        print('Number of valid metabolites with CCS data in full PPMI (y):', len(self.y))
        
        print('Number of metabolite feature columns (X):', self.X.shape[1])
        
    
    def save_state(self):
        filename = 'dataloader.p'        
        file_path = du.get_file_path(data_dir, 'class based structure', 'dataloaders', filename)
        du.dump_in_pickle(file_path, self)


#######################################################################################################################################################################

#######################################################################################################################################################################


class ExerciseMetabolomicsDataLoader():
    """This class can be used to obtain concentration change sign (CCS) data per metabolite for a specific publication. 
       This class can be further extended to generalize better to other publications
       Instructions need to be made for the input of custom data
    """
    def __init__(self, publication, calc_CCS_settings=None):
        #Based on the name of the publication, the correct source data files are identified and imported into the class object 
        self.source_data_files = self.get_source_file_locations(publication)
        self.blood, self.subjectFeatures, self.conditions = self.read_source_files()
        
        #Multiple experiments may be perfoemed on one subject in a metabolomics experiment and the identification of measuremnts and conditions is captured here 
        self.subjectID_conditionIDs = self.get_conditionIDs_per_subjectID()
        
        #Researchers may report metabolite names instead of metabolite IDs, these then need to be matched to IDs
        self.metabolite_names = self.blood.columns
        
        #The pre and post matrices contain abundance values per metabolite per subject 
        self.pre, self.post = self.get_pre_post_blood_values()
        
        #The log2fold change matrix contains log2fold change values per metabolite and per subject
        #A log2fold change value is easy to interpret, as a value of -1 means halving of abundance, and a value of 1 means a doubling in abundance.
        self.log2fold_change_matrix = self.calc_log2fold_change_matrix()
        
        #The log2fold change mtrix can be averaged across subjects to obtain the mean log2fold_change per metabolite
        self.log2fold_change = self.calc_log2fold_change()
        
        #Based on the mean log2fold_change, concentration change sign (CCS) can be obtained.
        #There are different methods of doing this. In the research, the naive method with p=0.1 is used for approximately equal class distribution. 
        #If no setting is specified the default setting from the research is used
        if type(calc_CCS_settings)==dict:
            self.calc_CCS_settings = calc_CCS_settings
        else:
            self.calc_CCS_settings = self.get_default_calc_CCS_settings()
        
        #Given the settings, the CCS values are obtained
        self.CCS = self.calc_CCS()
        
        
    def get_source_file_locations(self, publication):
        if publication=='millan':
            source_data_files = {'blood': du.get_file_path(data_dir, 'Exercise metabolomics DB', 'millan', 'millan_2020_blood.csv'),
                                 'conditions': du.get_file_path(data_dir, 'Exercise metabolomics DB', 'millan', 'millan2020_conditions.csv'),
                                 'subjectFeatures': du.get_file_path(data_dir, 'Exercise metabolomics DB', 'millan', 'millan2020_subjectFeatures.csv')}
        else:
            raise InputError(f"Incorrect publication name: {publication}")
            
        return source_data_files
    
    def read_source_files(self):
        blood = pd.read_csv(self.source_data_files['blood'], index_col = 'conditionID')
        subjectFeatures = pd.read_csv(self.source_data_files['subjectFeatures'])
        conditions = pd.read_csv(self.source_data_files['conditions'])
        return blood, subjectFeatures, conditions
        
    def get_conditionIDs_per_subjectID(self):
        conditions_subjectID = self.conditions.set_index('subjectID', drop=True)
        conditionsID_pre = conditions_subjectID[conditions_subjectID['pre_exercise_minutes']==1]['conditionID']
        conditionsID_post = conditions_subjectID[conditions_subjectID['post_exercise_minutes']==1]['conditionID']
        conditionsID_during = conditions_subjectID[conditions_subjectID['during_exercise_minutes']==1]['conditionID']

        subjectIDs = list(set(self.conditions['subjectID']))
        subjectIDs_int = [int(subjectID.split('_')[0].split('C')[1]) for subjectID in subjectIDs]
        
        subjectID_conditionIDs = pd.DataFrame([subjectIDs, conditionsID_pre[subjectIDs], np.full(len(subjectIDs), np.nan), conditionsID_post[subjectIDs], subjectIDs_int], \
                                              index=['subjectID', 'conditionID_pre', 'conditionID_during', 'conditionID_post', 'subjectIDs_int']).T.sort_values('subjectIDs_int')
        subjectID_conditionIDs = subjectID_conditionIDs.set_index('subjectID', drop=True).drop('subjectIDs_int', axis=1)
        return subjectID_conditionIDs
    
    def get_pre_post_blood_values(self):
        pre = self.blood.loc[list(self.subjectID_conditionIDs['conditionID_pre'])]
        post = self.blood.loc[list(self.subjectID_conditionIDs['conditionID_post'])]
        return pre, post
    
    def calc_log2fold_change_matrix(self):
        log2_pre = self.pre.reset_index(drop=True).astype(float).apply(np.log2).replace([np.inf, -np.inf], np.nan)
        log2_post = self.post.reset_index(drop=True).astype(float).apply(np.log2).replace([np.inf, -np.inf], np.nan)
        log2fold_change_matrix = log2_post - log2_pre
        log2fold_change_matrix.index = self.subjectID_conditionIDs.index
        return log2fold_change_matrix
    
    def calc_log2fold_change(self):        
        return self.log2fold_change_matrix.mean()

    def calc_CCS_naive(self, value, p_value=0.1):
        if value > p_value:
            return 1
        elif value < -p_value:
            return -1
        else:
            return 0
    
    def calc_CCS(self): 
        if self.calc_CCS_settings['method']=='naive':
            CCS = self.log2fold_change.apply(self.calc_CCS_naive, args=(self.calc_CCS_settings['p_value'],))
        else:
            raise InputError(f"Incorrect CCS calculation method: {method}. Pick from ['naive']")
        
        return CCS

    def get_default_calc_CCS_settings(self):
        default_calc_CCS_settings = {'method': 'naive', 
                                     'p_value': 0.1}
        
        return default_calc_CCS_settings
    
    def get_CCS_distribution_stats(self):
        return pd.DataFrame([self.CCS.value_counts(), self.CCS.value_counts()/len(self.CCS)*100], index=['Amount', 'Percentage']).T.style.format({'Amount': "{:.0f}", "Percentage": "{:.1f}%"})
    
    def plot_changes(self, change_type, from_id = 0, till_id = 40, save_fig=False):
        if change_type=='Mean log2fold change':
            change = self.log2fold_change
        elif change_type=='CCS':
            change = self.CCS
        else:
            raise InputError(f"Incorrect change_type: {change_type}. Pick from ['Mean log2fold change', 'CCS']")
        df = pd.DataFrame([self.CCS.index, change], index=['Metabolite', change_type]).T
        df = df.set_index('Metabolite')

        fig = plt.figure(figsize=(3,8))
        fig.patch.set_facecolor('white')    
        plt.barh(df.index[from_id:till_id], df[change_type][from_id:till_id])
        plt.xlabel(change_type, fontsize=14)
        plt.tight_layout()
        if save_fig:            
            filename = f'sportomics--{change_type}--{from_id}--{till_id}.png'
            plt.savefig(Path('figures', filename), bbox_inches='tight')
        plt.show()

#######################################################################################################################################################################

#######################################################################################################################################################################

        
class MetaboliteNameMatcher():
    """This class can take in a list of metabolite names and will create a matrix of matched HMDB_IDs and KEGG_IDs
       All calculations are performed at intialisation. It uses multiple data components to perform the match:
          - hmdb_accession_synonyms: This is a list of dicts per HMDB metabolite. Each dict has HMDB ID, HMDB name and HMDB synonyms 
          - hmdb_secondary_accessions: This is a list of dicts per HMDB metabolite. Each dict has HMDB ID and HMDB secondary accessions.  
          - kegg_compounds: This is a list of dicts per KEGG compound. Each dict has compound names 
          - compound_cmpdID: This is a datafile specific to the San-Millan (2020) publication with known HMDB/KEGG IDs per metabolite name
          - kegg_to_hmdb: pandas DataFrame with 2 columns: KEGG ID and HMDB ID. Used to obtain unknown HMDB IDs for known KEGG compounds.
          
        The class can also be initialized with name_accessions and conversion_table already created but with some additional matching to be performed.
        The class could be further extended to deal with the hmdb_secondary_accessions too
    """
    def __init__(self, metabolite_names_list, name_accessions=None, conversion_table=None):
        #The Source files structure depends on the data_utils.py functionality
        #It assumes the data is stored in a "Data" folder as described in the README of this repository
        self.source_files = self.get_source_files()
        self.hmdb_accession_synonyms, self.hmdb_secondary_accessions, self.kegg_compounds, self.compound_cmpdID, self.kegg_to_hmdb = self.import_source_files()

        #The metabolite_names_list is the target list of strings to be converted to IDs
        self.metabolite_names_list = metabolite_names_list

        #The conversion table is a dictionary that can be used to match specific strings to specific known IDs/strings
        #The goal here is to keep the functionality general and applicable for other metabolite names lists too
        self.conversion_table = conversion_table
        
        #The name accessions is the target pandas DataFrame that we'd like to have as complete and filled as possible. 
        #Rows in the name_accessions df represent metabolite name strings to match, the first column is HMDB IDs and the second column is KEGG ID 
        if type(name_accessions) == pd.DataFrame:
            self.name_accessions = name_accessions
        else:
            self.name_accessions = self.obtain_name_accessions() 
                
        #Any row for which neither HMDB nor KEGG ID was found is an unnamed metabolite
        self.unnamed_metabolites = self.create_unnamed_metabolites_table()

        #For the San-Millan (2020) publication, HMDB or KEGG IDs are listed per metabolite string. Where required this knowledge is used to create a conversion table. 
        self.conversion_table = self.extend_conversion_table()
        
        #If there are items in the conversion table, the matching process is repeated with the conversion table included to optimize metabolite identification recall
        if len(self.conversion_table) > 0:
            self.name_accessions = self.obtain_name_accessions() 
            self.unnamed_metabolites = self.create_unnamed_metabolites_table()
        
        #In a final step the self.kegg_to_hmdb is used to impute all unknown HMDB IDs based on known KEGG IDs. This yields additional HMDB IDs.
        self.imput_hmdb_id_from_kegg()
            
    def get_source_files(self):
        hmdb_metabolites_synonyms_file = du.get_file_path(data_dir, 'HMDB metabolites', 'Parsed pickle', 'hmdb_metabolites_synonyms.p')
        hmdb_metabolites_secondary_accessions_file = du.get_file_path(data_dir, 'HMDB metabolites', 'Parsed pickle', 'hmdb_metabolites_secondary_accessions.p')
        kegg_compounds_pickle_file = du.get_file_path(data_dir, 'Kegg compounds', 'pickle', 'kegg_compounds.p')
        compound_cmpdID_file = du.get_file_path(data_dir, 'Exercise metabolomics DB', 'millan', 'compound_cmpdID.csv')
        metabolite_kegg_to_hmdb_file = du.get_file_path(data_dir, 'HMDB metabolites', 'Parsed pickle', 'kegg_id_to_hmdb_id.p')
        
        source_files = {'hmdb_synonyms': hmdb_metabolites_synonyms_file,
                        'hmdb_secondary_accessions': hmdb_metabolites_secondary_accessions_file,
                        'kegg_compounds': kegg_compounds_pickle_file,
                        'compound_cmpdID': compound_cmpdID_file,
                        'kegg_to_hmdb': metabolite_kegg_to_hmdb_file}
        return source_files
    
    def import_source_files(self):
        hmdb_accession_synonyms = du.read_from_pickle(self.source_files['hmdb_synonyms'])
        hmdb_secondary_accessions = du.read_from_pickle(self.source_files['hmdb_secondary_accessions'])
        kegg_compounds = du.read_from_pickle(self.source_files['kegg_compounds'])
        compound_cmpdID = pd.read_csv(self.source_files['compound_cmpdID'])
        kegg_to_hmdb = du.read_from_pickle(self.source_files['kegg_to_hmdb'])
        
        return hmdb_accession_synonyms, hmdb_secondary_accessions, kegg_compounds, compound_cmpdID, kegg_to_hmdb

    def obtain_name_accessions(self):
        """This function is important logic for this class:
        For each string metabolite name in self.metabolite_names_list it will try to obtain HMDB ID and KEGG ID independently. 
        The results per string, even if they are not found are stored in the name_accessions pandas DataFrame"""
        hmdb_accessions = [self.find_hmdb_accession(self.hmdb_accession_synonyms, name) for name in self.metabolite_names_list]
        kegg_accessions = [self.find_kegg_accession(self.kegg_compounds, name) for name in self.metabolite_names_list]
        kegg_accessions_stripped = [kegg_accession[4:] if kegg_accession != None else None for kegg_accession in list(kegg_accessions)]
        
        hmdb_id_from_kegg = pd.Series(kegg_accessions_stripped).apply(self.get_hmdb_id) 

        name_accessions = pd.DataFrame([self.metabolite_names_list, pd.Series(hmdb_accessions), pd.Series(kegg_accessions), pd.Series(kegg_accessions_stripped), pd.Series(hmdb_id_from_kegg)], index = ['name', 'hmdb_accession', 'kegg_accession', 'kegg_accessions_stripped', 'hmdb_id_from_kegg_id']).T
        return name_accessions

    def imput_hmdb_id_from_kegg(self):
        hmdb_kegg_hmdb = self.name_accessions[['hmdb_accession', 'hmdb_id_from_kegg_id']]
        no_hmdb_id = hmdb_kegg_hmdb[hmdb_kegg_hmdb['hmdb_accession'].isna()]
        impute_hmdb_id_from_kegg = no_hmdb_id[no_hmdb_id['hmdb_id_from_kegg_id'].notna()]
        impute_hmdb_id_from_kegg['hmdb_accession'] = impute_hmdb_id_from_kegg['hmdb_id_from_kegg_id']
        self.name_accessions.update(impute_hmdb_id_from_kegg)        
    
    def create_unnamed_metabolites_table(self):
        unnamed_metabolites = self.name_accessions[self.name_accessions['hmdb_accession'].isna() & self.name_accessions['kegg_accession'].isna()]
        unnamed_metabolites.loc[:,'Reason for exclusion'] = [self.reason_for_exclusion(unnamed_metabolite) for unnamed_metabolite in unnamed_metabolites['name']]
        return unnamed_metabolites
    
    def find_hmdb_accession(self, accession_synonyms, synonym):
        try:
            synonym = self.conversion_table[synonym]
        except:
            pass

        for metabolite in accession_synonyms:
            if str.lower(synonym) == str.lower(metabolite['name']):
                return metabolite['accession']

            if synonym==metabolite['accession']:
                return metabolite['accession']

        for metabolite in accession_synonyms:
            synonyms_lower = [str.lower(synonym) for synonym in metabolite['synonyms']]
            if str.lower(synonym) in synonyms_lower:
                return metabolite['accession']

        return None

    def find_kegg_accession(self, kegg_compounds, name):
        try:
            name = self.conversion_table[name]
        except:
            pass

        if '/' in name:
            splitted_names = name.split('/')
            for splitted_name in splitted_names:
                accession = self.find_kegg_accession(kegg_compounds, splitted_name)
                if not accession==None:
                    return accession

        for compound in kegg_compounds:
            if name==compound['accession'] or 'cpd:'+name==compound['accession']:
                return name

            names_lowercased = [str.strip(str.lower(name)) for name in compound['names']]

            # Remove "'" from strings
            for name_lowercase in names_lowercased:
                if "'" in name_lowercase:
                    names_lowercased.append(name_lowercase.replace("'", ""))

            # Replace "," with "-" in all strings
            for name_lowercase in names_lowercased:
                if "," in name_lowercase:
                    names_lowercased.append(name_lowercase.replace(",", "-"))

            if str.strip(str.lower(name)) in names_lowercased:
                return compound['accession']

        return None

    def get_hmdb_id(self, kegg_id):
        if kegg_id != None:
            try:
                hmdb_ids = self.kegg_to_hmdb.loc[kegg_id]
                if len(hmdb_ids)>=1:
                    return hmdb_ids[0]
                else:
                    return None
            except:
                return None
        else:
            return None
    
    def find_hmdb_metabolite(self, accession):
        for metabolite in self.hmdb_accession_synonyms:
            if metabolite['accession'] == accession:
                return metabolite
        return None

    def find_kegg_compound(self, accession):
        for kegg_compound in self.kegg_compounds:
            if 'cpd:'+ accession == kegg_compound['accession']:
                return kegg_compound
        return None

    def find_binary(self, value):
        if value == -1:
            return False
        else:
            return True

    def find_compounds_that_contain_str(self, string):
        return self.compound_cmpdID[self.compound_cmpdID['compound'].apply(str.find, args=(string,)).apply(self.find_binary)]

    def create_dict_pair(self, unnamed_metabolite):
        try:
            accession = self.find_compounds_that_contain_str(unnamed_metabolite)['CmpdID'].reset_index(drop=True).get(0)
            if type(accession)==float and math.isnan(accession):
                return None

            if 'lipid' in accession:
                return None

            if accession not in ['sum']:
                return unnamed_metabolite, self.obtain_name(accession)
            else:
                return None
        except:
            return None

    def reason_for_exclusion(self, unnamed_metabolite):
        if unnamed_metabolite == 'Tetradecenoylcarnitine':
            return 'HMDBID not known'

        if '.1' in unnamed_metabolite:
            return 'duplicate'

        try:
            accession = self.find_compounds_that_contain_str(unnamed_metabolite)['CmpdID'].reset_index(drop=True).get(0)
            if type(accession)==float and math.isnan(accession):
                return 'is not linked to CmpdID'

            if 'lipid' in accession:
                return 'is lipid'

            if accession in ['sum']:
                return 'is sum'
        except:
            return 'other error'

        return 'other reason'

    def obtain_kegg_synonym(self, accession):
        kegg_compound = self.find_kegg_compound(accession)
        return kegg_compound['names'][0]

    def obtain_hmdb_name(self, accession):
        metabolite = self.find_hmdb_metabolite(accession)
        return metabolite['name']    

    def obtain_name(self, accession):
        if 'C' in accession:
            return self.obtain_kegg_synonym(accession)
        elif 'HMDB' in accession:
            return self.obtain_hmdb_name(accession)
        else:
            return None

    def extend_conversion_table(self, conversion_table={}):
        dict_pairs = [self.create_dict_pair(unnamed_metabolite) for unnamed_metabolite in self.unnamed_metabolites['name']]
        dict_pairs = [pair for pair in dict_pairs if pair != None]

        for pair in dict_pairs:
            old, new = pair
            conversion_table[old] = new

        return conversion_table  

    def is_full_match(self, target_str, strings):
        full_match = True
        for string in strings:
            if not(str.lower(string) in str.lower(target_str)):
                full_match = False

        return full_match

    def find_hmdb_metabolites_that_contains_strings(self, accession_synonyms, strings):
        result = []
        for metabolite in accession_synonyms:
            if is_full_match(metabolite['name'], strings):
                result.append(metabolite)
                continue
            synonyms_lower = [str.lower(synonym) for synonym in metabolite['synonyms']]      
            for synonym in synonyms_lower:
                if is_full_match(str.lower(synonym), strings):
                    result.append(metabolite)
                    continue

        return result

    def create_empty_unnamed_metabolites(self):
        return pd.DataFrame([], columns = ['name', 'hmdb_accession', 'kegg_accession', 'Reason for exclusion'])

    def create_hmdb_name_accession(self, metabolites_hmdb):
        return metabolites_hmdb[['name', 'accession']].set_index('name', drop=True)

    def import_unnamed_metabolites(self, unnamed_metabolites_file):
        try:
            unnamed_metabolites = pd.read_csv(unnamed_metabolites_file)
        except:
            unnamed_metabolites = create_empty_unnamed_metabolites()
        return unnamed_metabolites

    def import_conversion_table(self, conversion_table_file):
        try:
            conversion_table = du.read_from_pickle(conversion_table_file)
        except:
            conversion_table = {}
        return conversion_table

    def create_final_table(self):
        unnamed_targets = self.unnamed_metabolites.set_index('name')
        name_accessions = self.name_accessions.set_index('name')
        final_table = name_accessions.drop(unnamed_targets.index)
        return final_table 
    
class InputError(Exception):
    pass
