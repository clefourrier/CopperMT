import itertools, os, datetime, sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from data.management import get_inheritance_tree, get_children_relations, ancestors

path_to_etymdb = "/Users/cfourrie/documents/almanach/software/public_versions/EtymDB_public/data/split_etymdb/"
out_path = "/Users/cfourrie/documents/almanach/software/cognate_analysis/inputs/raw_data"
langs_of_interest = sorted(["pt", "gl", "es", "ca", "oc", "it", "fr", "ro", "rup", "dlm", "la"]) #
data_name = "romance_bilingual"

# Sources in the file next to this one
path_values = os.path.join(path_to_etymdb, "etymdb_values.csv")
path_link = os.path.join(path_to_etymdb, "etymdb_links_info.csv")

df_values = pd.read_csv(path_values,
                        sep='\t',
                        names=["id", "lang", "field", "lexeme", "meaning"],
                        dtype={"id": int, "lang": str, "field": int, "meaning": str}).set_index("id")
df_link = pd.read_csv(path_link,
                      sep='\t',
                      names=["relation_type", "child", "parent"],
                      dtype={"relation_type": str, "child": int, "parent": int})

df_inher = df_link.loc[df_link['relation_type'].isin(["inh"])]
print("Data read")

# Inheritance data
langs_of_interest_and_their_parents = []
for l in langs_of_interest:
    langs_of_interest_and_their_parents.extend(ancestors[l] + [l])
langs_of_interest_and_their_parents = list(set(langs_of_interest_and_their_parents))
print(langs_of_interest_and_their_parents)

inher_dag = get_inheritance_tree(df_inher, df_values,
                                 langs_of_interest_and_their_parents)
print("Global inheritance extracted")
cog_set = get_children_relations(inher_dag, langs_of_interest, ancestors)
print("Children relations extracted")

cog_set.clean()
print("Cognate set cleaned")


# Cognates values
folder_path = os.path.join(out_path, f"{data_name}_{datetime.datetime.now().date()}")
try:
    os.mkdir(folder_path)
except FileExistsError:
    pass
for l_in, l_out in itertools.product(langs_of_interest, langs_of_interest):
    cog_set.save(folder_path, [l_in, l_out])