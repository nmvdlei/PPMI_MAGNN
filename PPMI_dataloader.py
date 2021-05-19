import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx

import warnings
warnings.filterwarnings("ignore")

import data_utils as du
data_dir = du.find_data_dir('app')

class InputError(Exception):
    pass

class ExerciseMetabolomicsDataLoader():
    def __init__(self, publication, calc_CCS_settings=None):        
        self.source_data_files = self.get_source_file_locations(publication)
        self.blood, self.subjectFeatures, self.conditions = self.read_source_files()
        self.subjectID_conditionIDs = self.get_conditionIDs_per_subjectID()
        
        self.metabolite_names = self.blood.columns
        
        self.pre, self.post = self.get_pre_post_blood_values()
        self.log2fold_change_matrix = self.calc_log2fold_change_matrix()
        self.log2fold_change = self.calc_log2fold_change()
        
        if type(calc_CCS_settings)==dict:
            self.calc_CCS_settings = calc_CCS_settings
        else:
            self.calc_CCS_settings = self.get_default_calc_CCS_settings()
        
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
    
    def plot_changes(self, change_type, from_id = 0, till_id = 40):
        if change_type=='Mean log2fold change':
            change = self.log2fold_change
        elif change_type=='CCS':
            change = self.CCS
        else:
            raise InputError(f"Incorrect change_type: {change_type}. Pick from ['Mean log2fold change', 'CCS']")
        df = pd.DataFrame([self.CCS.index, change], index=['Metabolite', change_type]).T
        df = df.set_index('Metabolite')

        plt.figure(figsize=(3,8))
        plt.barh(df.index[from_id:till_id], df[change_type][from_id:till_id])
        plt.xlabel(change_type, fontsize=14)
        plt.show()
    
    
# millan_2020 = ExerciseMetabolomicsDataLoader('millan')
# millan_2020.CCS

class MetaboliteNameMatcher():
    def __init__(self, metabolite_names_list, name_accessions=None, conversion_table=None):
        self.source_files = self.get_source_files()
        self.hmdb_accession_synonyms, self.kegg_compounds, self.compound_cmpdID = self.import_source_files()

        self.metabolite_names_list = metabolite_names_list
        self.conversion_table = conversion_table        
        if type(name_accessions) == pd.DataFrame:
            self.name_accessions = name_accessions
        else:
            self.name_accessions = self.obtain_name_accessions() 
        
        self.unnamed_metabolites = self.create_unnamed_metabolites_table()
        
        self.conversion_table = self.extend_conversion_table()
        
        if len(self.conversion_table) > 0:
            self.name_accessions = self.obtain_name_accessions() 
            self.unnamed_metabolites = self.create_unnamed_metabolites_table()
        
    def get_source_files(self):
        hmdb_metabolites_synonyms_file = du.get_file_path(data_dir, 'HMDB metabolites', 'Parsed pickle', 'hmdb_metabolites_synonyms.p')
        kegg_compounds_pickle_file = du.get_file_path(data_dir, 'Kegg compounds', 'pickle', 'kegg_compounds.p')

        compound_cmpdID_file = du.get_file_path(data_dir, 'Exercise metabolomics DB', 'millan', 'compound_cmpdID.csv')
        
        source_files = {'hmdb_synonyms': hmdb_metabolites_synonyms_file,
                        'kegg_compounds': kegg_compounds_pickle_file,
                        'compound_cmpdID': compound_cmpdID_file}
        return source_files
    
    def import_source_files(self):
        hmdb_accession_synonyms = du.read_from_pickle(self.source_files['hmdb_synonyms'])
        kegg_compounds = du.read_from_pickle(self.source_files['kegg_compounds'])
        compound_cmpdID = pd.read_csv(self.source_files['compound_cmpdID'])
        
        return hmdb_accession_synonyms, kegg_compounds, compound_cmpdID

    def obtain_name_accessions(self):
        hmdb_accessions = [self.find_hmdb_accession(self.hmdb_accession_synonyms, name) for name in self.metabolite_names_list]
        kegg_accessions = [self.find_kegg_accession(self.kegg_compounds, name) for name in self.metabolite_names_list]

        return pd.DataFrame([self.metabolite_names_list, pd.Series(hmdb_accessions), pd.Series(kegg_accessions)], index = ['name', 'hmdb_accession', 'kegg_accession']).T

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
    
# millan_exercise_metabolomics_file = du.get_file_path(data_dir, 'Exercise metabolomics DB', 'millan', 'metabolites_change.csv')
# millan_exercise_metabolomics = pd.read_csv(millan_exercise_metabolomics_file)
# matcher = MetaboliteNameMatcher(millan_exercise_metabolomics['Metabolite'])

class DataLoader():
    def __init__(self, include_feature_category, metabolite_matcher_file=None, calc_CCS_settings=None, allowed_ccs_values=[-1, 0, 1]):
        self.include_feature_category = include_feature_category
        self.allowed_ccs_values = allowed_ccs_values
        self.millan_2020 = ExerciseMetabolomicsDataLoader('millan', calc_CCS_settings)
        
        if metabolite_matcher_file is None:
            metabolite_matcher_file = du.get_file_path(data_dir, 'class based structure', 'metabolite matching', 'matcher.p')
            self.metabolite_matcher = du.read_from_pickle(metabolite_matcher_file)
        else:
            self.metabolite_matcher = MetaboliteNameMatcher(list(self.millan_2020.log2fold_change.index))
            
        self.hmdb_log2fold_change_CSS = self.get_hmdb_metabolites_with_data()
        self.hmdb_log2fold_change_CSS = self.get_non_duplicate_rows(self.hmdb_log2fold_change_CSS)
        self.hmdb_log2fold_change_CSS = self.get_rows_with_allowed_values(self.hmdb_log2fold_change_CSS)
        
        networkx_pickle_file = du.get_file_path(data_dir, 'interactome', 'networkx pickle', 'prot_hmdb_networkx_graph.p')
        self.PPMI_full = nx.read_gpickle(networkx_pickle_file)
        self.remove_isolated_nodes_from_full_PPMI()
            
        self.hmdb_log2fold_change_CSS = self.get_metabolites_in_PPMI(self.hmdb_log2fold_change_CSS)
        
        self.PPMI_pruned = self.prune_PPMI_network(self.hmdb_log2fold_change_CSS.index)
        self.remove_isolated_nodes_from_pruned_PPMI()
        
        self.y = self.hmdb_log2fold_change_CSS['CCS']
        self.X = self.construct_metabolite_feature_df()
        
        
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
        
    def get_metabolite_nodes(self):
        all_nodes = list(self.PPMI_full.nodes)
        metabolites_in_PPMI = [m for m in all_nodes if 'HMDB' in m]
        return metabolites_in_PPMI
    
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
            
    def construct_metabolite_feature_df(self):
        metabolite_features_dfs_list = [] 

        for feature_name, include in self.include_feature_category.items():
            if include:
                file_name = du.get_file_path(data_dir, 'HMDB metabolites', 'Feature dfs pickle', f'hmdb_metabolites_{feature_name}.p')
                feature_df = du.read_from_pickle(file_name)
                metabolite_features_dfs_list.append(feature_df)

        metabolite_features_all_df = pd.concat(metabolite_features_dfs_list, axis=1)        
        metabolite_features_df = metabolite_features_all_df.loc[self.hmdb_log2fold_change_CSS.index]
        
        #Remove those columns with only a single unique value
        columns_with_single_unique_value = metabolite_features_df.columns[metabolite_features_df.nunique()==1]
        metabolite_features_df = metabolite_features_df.drop(columns_with_single_unique_value, axis=1)
        
        return metabolite_features_df
    
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
        nodes_to_remove = ['HMDB0000562', 'HMDB0001036', 'AOPEP', 'SLC47A1', 'SLC47A2', 'PRHOXNB']

        for node in nodes_to_remove:
            if node in self.PPMI_full.nodes:
                self.PPMI_full.remove_node(node)
    
    def remove_isolated_nodes_from_pruned_PPMI(self):
        nodes_to_remove = ['ALLC', 'HMDB0001209']

        for node in nodes_to_remove:
            if node in self.PPMI_pruned.nodes:
                self.PPMI_pruned.remove_node(node)
                if 'HMDB' in node:
                    self.hmdb_log2fold_change_CSS = self.hmdb_log2fold_change_CSS.drop(node)
                    
    def print_components(self, G):
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
        

# calc_CCS_settings = {'method': 'naive', 
#                      'p_value': 0.1}

# allowed_ccs_values = [-1, 0, 1]

# include_feature_category = {'molecular_weight': True,
#                             'state': True,
#                             'kingdom': True,
#                             'super_class': True,
#                             'class': True,
#                             'direct_parent': True,
#                             'molecular_framework': True,
#                             'alternative_parents': True,
#                             'substituents': True,
#                             'external_descriptors': True,
#                             'cellular_locations': True,
#                             'biospecimen_locations': True,
#                             'tissue_locations': True}
    
# dataloader = DataLoader(include_feature_category, calc_CCS_settings=calc_CCS_settings, allowed_ccs_values=allowed_ccs_values)
# dataloader.print_data_counts()
# dataloader.target_distribution_stats()
# dataloader.save_state()