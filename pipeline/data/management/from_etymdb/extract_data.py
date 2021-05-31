import networkx as nx
from collections import defaultdict
from .utils.word_sets import WordSetCollection


def get_inheritance_tree(df, df_values, langs_of_interest = None):
    """ Constructs a inheritance tree with all items from etymdb"""
    def add_node(inher_tree, ix, df_values):
        inher_tree.add_node(ix)
        nx.set_node_attributes(inher_tree,
                               {ix: {"lang": df_values.loc[ix].lang,
                                     "lexeme": df_values.loc[ix].lexeme,
                                     "meaning": df_values.loc[ix].meaning}}
                               )
    inher_tree = nx.DiGraph()
    for index, row in df.iterrows():
        child_ix = row["child"]
        parent_ix = row["parent"]
        if child_ix >= 0 and parent_ix >= 0:
            pair_ok = not langs_of_interest or (
                    df_values.loc[parent_ix].lang in langs_of_interest and
                    df_values.loc[child_ix].lang in langs_of_interest
            )
            if pair_ok:
                add_node(inher_tree, parent_ix, df_values)
                add_node(inher_tree, child_ix, df_values)
                inher_tree.add_edge(parent_ix, child_ix)

    return inher_tree


def get_children_relations(dag: nx.DiGraph, children_langs: list,
                           allowed_ancestors: dict) -> WordSetCollection:
    """ Extends the cog_set and bor_set with cognates and borrowings found exploring the dag
    Only use this function when you are dealing with a Directed Acyclic Graph

    :param dag: Directional Acyclic Graph containing the inheritance links
    :param allowed_ancestors: Dict of allowed ancestors for a given language
    If the ancestors are allowed, it will go in the cogset, else in the borset
    :return: The modified cognates and borrowing sets
    """
    def is_sublist(sublst, lst):
        for element in sublst:
            try:
                ind = lst.index(element)
            except ValueError:
                return False
            lst = lst[ind + 1:]
        return True
    cog_set = WordSetCollection()

    # We create a set of all possible ancestor languages
    parent_langs = []
    for child_lang in children_langs:
        parent_langs.extend(allowed_ancestors[child_lang])
    parent_langs = set(parent_langs)

    # We look at the nodes without parents (roots of trees)
    for deg in range(max(d for n, d in dag.in_degree()) - 1):
        for source in [n for n, d in dag.in_degree() if d == deg and dag.nodes[n]["lang"] in parent_langs]:
            # We save the parent word
            cur_cogs = defaultdict(list)
            cur_cogs[dag.nodes[source]["lang"]].append(dag.nodes[source]["lexeme"])

            # We look at the descendants of the correct languages
            all_descendants = nx.descendants(dag, source)
            target_descendants = [d for d in all_descendants if dag.nodes[d]["lang"] in children_langs]
            if len(target_descendants) == 0:
                continue  # If no descendants, we continue

            # If we have two items or more that are "cousins"
            for target in target_descendants:
                target_lang = dag.nodes[target]["lang"]
                paths_source_to_target = nx.all_simple_paths(dag, source=source, target=target)
                # We look at each path (we should not get more than one often)
                for path in paths_source_to_target:
                    # If all ancestors are allowed and ordered properly
                    if is_sublist([dag.nodes[ix]["lang"] for ix in path], allowed_ancestors[target_lang] + [target_lang]):
                        cur_cogs[target_lang].append(dag.nodes[target]["lexeme"])

            if len(cur_cogs.keys()) > 1:
                cog_set.add_word_set(cur_cogs)

    return cog_set
