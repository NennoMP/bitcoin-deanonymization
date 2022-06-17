import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def gen_pairs(df1: pd.DataFrame, df2: pd.DataFrame, pair_column: str, merge_column: str) -> pd.DataFrame:
    """Creates pair-wise combinations of <pair_column> sharing at least one <merge_column> value. Between two dataframes or in the dataframe itself if df1 == df2.

    :param <df1>: first dataframe
    :param <df2>: second dataframe
    :param <pair_column>: column name for which to create pairs
    :param <merge_column>: column name on which the merge is based

    :return: pandas dataframe of pairs in edge format (x -> y)
    """

    # Labels of the pairs
    x_label = f'{pair_column}_x'
    y_label = f'{pair_column}_y'

    # Generate pairs
    nwk = df1[[pair_column, merge_column]].merge(df2[[pair_column, merge_column]], on=merge_column).drop(columns=merge_column)

    # Drop duplicate pairs (undirected edges)
    nwk = nwk.loc[pd.DataFrame(np.sort(nwk[[x_label, y_label]],1),index=nwk.index).drop_duplicates(keep='first').index]

    return nwk


def gen_new_pairs(df: pd.DataFrame, cum_inputs: pd.DataFrame, cum_pairs: pd.DataFrame) -> pd.DataFrame:
    """Creates new pair-wise combinations of <tx_id> sharing at least one <address> value.

    :param <df>: inputs of next K blocks
    :param <cum_inputs>: inputs found in previous time-steps
    :param <cum_pairs>: pairs found in previous time-step

    :return: pandas dataframe of new pairs in edge format (x -> y)
    """

    # New pairs within the new group of <K> blocks
    nwk = gen_pairs(df, df, pair_column='tx_id', merge_column='address')

    # New pairs between new group of <K> blocks and previous blocks
    nwk = pd.concat([gen_pairs(cum_inputs, df, pair_column='tx_id', merge_column='address'), nwk], ignore_index=True, sort=False)

    # Concatenate new pairs with previous ones
    nwk = pd.concat([cum_pairs, nwk], ignore_index=True, sort=False)

    # Drop duplicate pairs (undirected edges)
    nwk = nwk.loc[pd.DataFrame(np.sort(nwk[['tx_id_x', 'tx_id_y']],1),index=nwk.index).drop_duplicates(keep='first').index]

    return nwk


def connected_components(pairs: pd.DataFrame, source: str, target: str) -> list[list]:
    """Computes clusters/groups leveraging the connected components property of a graph.

    :param <pairs>: pairs of edges
    :param <source>: column name in <pairs> to use as source
    :param <target>: column name in <pairs> to use as target

    :return: list of list of clusters
    """

    # Create the graph from the pairs (edges)
    G = nx.from_pandas_edgelist(pairs, source=source, target=target)

    for c in nx.connected_components(G):
        yield list(G.subgraph(c))


def label_clusters(df: pd.DataFrame, clusters, diff=None) -> pd.DataFrame:
    """Labels each cluster with an id in the original dataframe and adds clusters of size 1 with output addresses encountered only once.

    :param <df>: original dataframe
    :param <clusters>: list of list of clusters (addresses) found
    :param <diff>: output addresses encountered only once (will form clusters of size 1)

    :return: pandas dataframe with cluster id for each address
    """

    # Uniquely label each cluster
    d = {k: i for i in range(len(clusters)) for k in clusters[i]}

    # Bring label back to original DataFrame
    df['cluster_id'] = df['tx_id'].map(d)

    # Now add unique addresses from outputs
    # Get counter of cluster_id
    cluster_count = df.cluster_id.iloc[-1] + 1

    # Add single clusters of output addresses to a dataframe
    tmp = pd.DataFrame(
        [[v, cluster_count+i] for i, v in enumerate(diff)], columns=['address', 'cluster_id'])

    # Concatenate with main dataframe
    df = pd.concat([df, tmp], ignore_index=True, sort=False)

    # Remove duplicate pairs (for transactions with multiple inputs with same address)
    df = df.loc[pd.DataFrame(np.sort(df[['address','cluster_id']],1),index=df.index).drop_duplicates(keep='first').index]

    return df


def plot_scatter(xdata, ydata, filename: str, title: str, xlabel: str, ylabel: str, color='#2986cc'):
    """Plot and save a scatter (distribution) chart."""

    plt.scatter(xdata, ydata, alpha=0.5, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, linestyle='--')
    plt.savefig(filename)
    plt.clf()


def plot_trend(xdata, ydata, filename: str, title: str, xlabel: str, ylabel: str, marker=None, color='#2986cc'):
    """Plot and save a temporal (trend) chart."""

    plt.plot(xdata, ydata, marker=marker, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, linestyle='--')
    plt.savefig(filename)
    plt.clf()
