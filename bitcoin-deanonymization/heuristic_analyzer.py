import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

import utils

# -------------------------
# LOGGING CONFIGURATION
# -------------------------
LOG_FORMAT = '%(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO
LOG_FILE = 'bitcoin-deanonymization/data/logs/heuristic_effectiveness.log'
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',
    format=LOG_FORMAT,
    level=LOG_LEVEL)

# Plot files path
PLOTS_PATH = r'bitcoin-deanonymization/data/plots/'


class HeuristicAnalyzer:
    """Class implementing heuristics logic and helpers for plotting analyses"""

    # Dataframes
    tx_df: pd.DataFrame
    in_df: pd.DataFrame
    out_df: pd.DataFrame

    addresses_set_size: int
    K: int

    # OCH Attributes
    occurrences: set # appeared addresses
    cluster_id: int

    # Dataframes for cumulative data (each timestep)
    cum_inputs: pd.DataFrame
    cum_outputs: pd.DataFrame
    cum_pairs: pd.DataFrame
    cum_clusters: pd.DataFrame

    # Dictionaries for data over time (<n_blocks>: <data>)
    temporal_reduction_mih: dict
    temporal_addresses_size: dict
    temporal_clusters_mih: dict
    temporal_clusters_och: dict

    def __init__(self, tx_df: pd.DataFrame, in_df: pd.DataFrame, out_df: pd.DataFrame, K: int):
        """
        :param <tx_df>: transactions dataframe
        :param <in_df>: inputs dataframe
        :param <out_df>: outputs dataframe
        :param <K>: # of blocks (time-step) for temporal analysis
        """

        self.tx_df = tx_df
        self.in_df = in_df
        self.out_df = out_df

        self.K = K

        self.occurrences = set()
        self.cluster_id = 0

        self.temporal_reduction_mih_mih = {}
        self.temporal_addresses_size = {}
        self.temporal_clusters_mih = {}
        self.temporal_clusters_och = {}

    def evaluate_heuristic(self, A: int, C: int) -> int:
        """Evaluates an heuristic effectiveness (reduction %) R according to the formula:

        R = (|A| - |C|) / |A|
       
        :param <A>: set of all addresses in the transactions
        :param <C>: set of all clusters after clustering

        :return: heuristic effectiveness (reduction %)
        """
        return (A - C) / A

    def plot_clusters_distribution(self, clusters: pd.DataFrame, heuristic: str, color='#2986cc'):
        """Plots the distribution of the clusters size after applying a specific heuristic.

        :param <clusters>: clusters of addresses
        :param <heuristic>: heuristic name
        """

        clusters_size = clusters.groupby('cluster_id')['cluster_id'].count()

        # Plot address clusters size distribution
        clusters_size_distribution = {}
        for size in clusters_size.iteritems():
            value = size[1]
            if value in clusters_size_distribution:
                clusters_size_distribution[value] += 1
            else:
                clusters_size_distribution[value] = 1

        utils.plot_scatter(
            xdata=clusters_size_distribution.keys(),
            ydata=clusters_size_distribution.values(),
            filename=f'{PLOTS_PATH}{heuristic}_clusters_distribution',
            title=heuristic,
            xlabel='Cluster size (# of addresses)',
            ylabel='Frequency',
            color=color
        )

    def plot_reduction_trend(self, clusters: pd.DataFrame, temporal_reduction: dict, title: str, heuristic: str, color='#2986cc'):
        """Plots the temporal reduction of addresses after applying a specific heuristic.

        :param <clusters>: clusters of addresses
        :param <temporal_reduction>: dictionary of reduction at each time-step
        :param <heuristic>: heuristic name
        """

        clusters_set_size = len(clusters.groupby('cluster_id').address.agg(set))

        # Compute effectiveness (reduction)
        effectiveness = self.evaluate_heuristic(
            A=self.addresses_set_size,
            C=clusters_set_size)
          
        logging.info(
            f'[{heuristic}-heuristic] - effectiveness {effectiveness}')

        # Plot reduction temporal trend
        utils.plot_trend(
            xdata=temporal_reduction.keys(),
            ydata=temporal_reduction.values(),
            filename=f'{PLOTS_PATH}{heuristic}_trend_{self.K}',
            title=title,
            xlabel='Blocks',
            ylabel='Reduction',
            marker='.',
            color=color
        )

    def plot_improvement_trend(self, improvement: dict, title: str, xlabel: str, ylabel: str, filename: str, color='black'):
        """Plots the temporal improvement of addresses reduction, comparing MIH and MIH+OCH.

        :param <improvement>: addresses reduction ratio improvement trend after applying MIH+OCH 
        """

        # Plot improvement
        plt.plot(improvement.keys(), improvement.values(), marker='.', color=color, label='Improvement')

        # Plot average improvement line
        y_avg = [np.mean(list(improvement.values()))] * len(improvement.keys())
        plt.plot(improvement.keys(), y_avg, color='red', lw=2, ls='--', label="Average")

        plt.legend()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.grid(True, linestyle='--')
        plt.savefig(filename)
        plt.clf()

    def init_cumulative_dataframes(self):
        """Initializes/Resets cumulative dataframes before applying an heuristic, since they're shared attributes.
        """

        self.cum_inputs = pd.DataFrame(columns=['block_id', 'tx_id', 'address'])
        self.cum_outputs = pd.DataFrame(columns=['block_id', 'tx_id', 'address'])
        self.cum_pairs = pd.DataFrame(columns=['tx_id_x', 'tx_id_y'])
        self.cum_clusters = pd.DataFrame(columns=['address', 'cluster_id'])

    def get_blocks_groups(self):
        """Groups inputs and outputs by <self.K> blocks and returns them, used for temporal analyses.

        :return: two groupby dataframes (inputs and outputs) on K blocks
        """
        # Merge transactions - inputs
        txs_inputs = pd.merge(
            self.tx_df.set_index('id'), 
            self.in_df, 
            left_index=True, 
            right_on=['tx_id']
        )
        
        # Merge transactions - outputs
        txs_outputs = pd.merge(
            self.tx_df.set_index('id'), 
            self.out_df, 
            left_index=True, 
            right_on=['tx_id']
        )

        # Group by <K> blocks
        in_groups = txs_inputs.groupby(txs_inputs.block_id // self.K)
        out_groups = txs_outputs.groupby(txs_outputs.block_id // self.K)

        return in_groups, out_groups

    def mih(self):
        """Computes Multi Input Addresses Heuristic (MIH) and related analyses."""

        # Init/Resets cumulative dataframes
        self.init_cumulative_dataframes()

        # Group inputs-outputs by <self.K> blocks
        in_groups, out_groups = self.get_blocks_groups()

        temporal_reduction = {}
        step = self.K

        for ((i_name, i_group), (o_name, o_group)) in zip(in_groups, out_groups):

            # Apply MIH considering the new blocks
            self.mih_one_step(i_group, o_group)

            # Drop non-needed columns
            self.cum_clusters.drop(columns=['block_id', 'tx_id'], inplace=True)

            # Get clusters set size (# clusters)
            clusters_set_size = len(self.cum_clusters.groupby('cluster_id').address.agg(set))

            # Get reduction (effectiveness) at this time step
            effectiveness = self.evaluate_heuristic(
            A=self.addresses_set_size,
            C=clusters_set_size)

            # Store data of current time-step
            temporal_reduction[step] = effectiveness
            self.temporal_addresses_size[step] = self.addresses_set_size
            self.temporal_clusters_mih[step] = self.cum_clusters.copy()

            self.temporal_reduction_mih = temporal_reduction

            step += self.K

        self.plot_clusters_distribution(self.cum_clusters, heuristic='MIH')

        self.plot_reduction_trend(
            self.cum_clusters,
            temporal_reduction,
            title=f'timestep = {self.K:_} (blocks)',
            heuristic='MIH'
        )

    def mih_one_step(self, i_group: pd.DataFrame, o_group: pd.DataFrame):
        """Computes one temporal step of MIH considering additional K transactions and previous ones if any. 
          
        :param <i_group>: new K blocks inputs
        :param <o_group>: new K blocks outputs
        """

        # Drop coinbase transactions in inputs group
        i_group = i_group.drop(i_group.index[i_group['address'] == 0])

        # Drop transactions with only one input sharing <address>
        # keep first occurence to avoid losing information
        one_input_txs = i_group[~i_group.index.isin(i_group[i_group.tx_id.duplicated(keep=False)].index)]['tx_id']
        i_group = i_group[(~i_group.address.duplicated())|~i_group.tx_id.isin(one_input_txs)]

        # Create transactions pair-wise combinations in edge graph format considering new blocks
        # <tx_id_x> to <tx_id_y> if two transactions share at least one address
        df_pairs = utils.gen_new_pairs(i_group, self.cum_inputs, self.cum_pairs)

        # Update cumulative data
        self.cum_inputs = pd.concat([self.cum_inputs, i_group], ignore_index=False, sort=False)
        self.cum_outputs = pd.concat([self.cum_outputs, o_group], ignore_index=False, sort=False)
        self.cum_pairs = df_pairs.copy()

        # Get addresses set size (# unique addresses)
        self.addresses_set_size = len(np.union1d(list(self.cum_inputs.address.unique()), list(self.cum_outputs.address.unique())))

        # Get unique addresses in outputs 
        # (i.e. that never appear in inputs)
        # these will form clusters of size 1
        diff = np.setdiff1d(list(self.cum_outputs.address.unique()), list(self.cum_inputs.address.unique()))

        # Create clusters of addresses by creating a graph with the <df_pairs> 
        # and computing the connected components of such graph
        groups = [*utils.connected_components(df_pairs, source='tx_id_x', target='tx_id_y')]

        # Bring back the clusters ids to the dataframe
        self.cum_clusters = utils.label_clusters(self.cum_inputs, groups, diff)

    def och(self):
        """Computes One-time Change Addresses Heuristic (OCH) and related analyses.
        """

        # Init/Resets cumulative dataframes
        self.init_cumulative_dataframes()

        # Group inputs-outputs by <self.K> blocks
        in_groups, out_groups = self.get_blocks_groups()

        # Get coinbase txs
        coinbase_txs = set(self.in_df[self.in_df['address'] == 0]['tx_id'])
        step = self.K

        for ((i_name, i_group), (o_name, o_group)) in zip(in_groups, out_groups):

            # Apply OCH considering the new blocks
            self.och_one_step(i_group, o_group, coinbase_txs)

            # Store data of current time-step
            #self.temporal_addresses_size[step] = self.addresses_set_size
            self.temporal_clusters_och[step] = self.cum_clusters.copy()

            step += self.K

    def och_one_step(self, i_group: pd.DataFrame, o_group: pd.DataFrame, coinbase_txs: pd.DataFrame):
        """Computes one temporal step of OCH considering additional K transactions and previous ones if any.
          
        :param <i_group>: new K blocks inputs
        :param <o_group>: new K blocks outputs
        """

        # Update cumulative data
        self.cum_inputs = pd.concat([self.cum_inputs, i_group], ignore_index=False, sort=False)
        self.cum_outputs = pd.concat([self.cum_outputs, o_group], ignore_index=False, sort=False)

        # Get addresses set size (# unique addresses)
        self.addresses_set_size = len(np.union1d(list(self.cum_inputs.address.unique()), list(self.cum_outputs.address.unique())))

        # Group by <tx_id>
        i_group = i_group.groupby('tx_id')
        o_group = o_group.groupby('tx_id')

        for ((i_name, i_group), (o_name, o_group)) in zip(i_group, o_group):
            # v[0]: index | v[1]: block_id | v[2]: tx_id | v[3]: address

            candidates = []
            inputs_addresses = list(i_group.address.values)

            # Check if coinbase or only one outputs address
            if o_name not in coinbase_txs and o_group.address.nunique() > 1:

                for v in o_group.itertuples():
                    # Check if output address appears in inputs
                    if v[3] in inputs_addresses:
                        candidates = []
                        break
                    
                    # Check if outputs address appears for the first time
                    if v[3] not in self.occurrences:
                        # Add possible one-time change address
                        candidates.append(v[3])

            # Store outputs addresses occurrences
            for v in o_group.itertuples():
                self.occurrences.add(v[3])
            # Store inputs addresses occurrences
            for el in list(inputs_addresses):
                self.occurrences.add(el)

            # If no output address appeared in inputs and only one candidate
            # create clusters (inputs + one-time change address found)
            if len(candidates) == 1:
                cluster = candidates + inputs_addresses

                # Create cluster
                tmp = pd.DataFrame(
                    [[address, self.cluster_id] for address in cluster], columns=['address', 'cluster_id'])

                # Update cumulative clusters
                self.cum_clusters = pd.concat([self.cum_clusters, tmp], ignore_index=True, sort=False)
                self.cluster_id += 1

    def mih_and_och(self):
        """Computes MIH and OCH sequentially (MIH + OCH heuristic) and related analyses.
        """

        temporal_reduction = {}
        for (mih_item, och_item) in zip(self.temporal_clusters_mih.items(), self.temporal_clusters_och.items()):

            key = mih_item[0]
            # key = och_item[0] is identical

            # Get clusters of current time-step
            mih_clusters = mih_item[1]
            och_clusters = och_item[1]

            # Get last cluster id in MIH clusters and update OCH cluster ids
            # to avoid conflicting cluster ids before intersection
            cluster_id = mih_clusters.cluster_id.iloc[-1] + 1
            och_clusters['cluster_id'] += cluster_id

            # Concatenate clusters of MIH and OCH
            tot_clusters = pd.concat([mih_clusters, och_clusters], ignore_index=True, sort=False)

            # Generate pairs (cluster intersections)
            df_pairs = utils.gen_pairs(
                tot_clusters,
                tot_clusters,
                pair_column='cluster_id',
                merge_column='address'
            )

            # Get connected components in the graph
            groups = [*utils.connected_components(
                df_pairs,
                source='cluster_id_x',
                target='cluster_id_y')]

            # Uniquely label each group
            d = {key: i for i in range(len(groups)) for key in groups[i]}

            # Bring label back to original dataframe
            tot_clusters['cluster_id'] = tot_clusters['cluster_id'].map(d)

            # Remove duplicate pairs
            # for transactions with multiple inputs with same address
            tot_clusters = tot_clusters.loc[pd.DataFrame(np.sort(tot_clusters[['address','cluster_id']],1),index=tot_clusters.index).drop_duplicates(keep='first').index]

            tmp_clusters = tot_clusters.copy()

            # Get clusters set size of current time-step
            tmp_clusters = tmp_clusters.groupby('cluster_id').address.agg(set)
            clusters_set_size = len(tmp_clusters)

            # Get effectiveness of current time-step
            effectiveness = self.evaluate_heuristic(
            A=self.temporal_addresses_size[key],
            C=clusters_set_size)
            temporal_reduction[key] = effectiveness


        self.plot_clusters_distribution(tot_clusters, heuristic='MIH+OCH', color='#f2a900')

        self.plot_reduction_trend(
            tot_clusters,
            temporal_reduction,
            title=f'timestep = {self.K:_} (blocks)',
            heuristic='MIH+OCH',
            color='#f2a900'
        )

        # Plot trends together
        plt.plot(self.temporal_reduction_mih.keys(), self.temporal_reduction_mih.values(), marker='.', label='MIH')
        plt.plot(temporal_reduction.keys(), temporal_reduction.values(), marker='.', label='MIH + OCH', color='#f2a900')

        plt.legend()
        plt.title(f'timestep = {self.K:_} (blocks)')
        plt.xlabel('Blocks')
        plt.ylabel('Reduction')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.grid(True, linestyle='--')

        plt.savefig(f'bitcoin-deanonymization/data/plots/all_trends_{self.K}')
        plt.clf()

        # Plot improvement trend
        improvement = {k: temporal_reduction[k] - self.temporal_reduction_mih[k] for k in self.temporal_reduction_mih.keys()}

        self.plot_improvement_trend(
            improvement=improvement,
            title=f'timestep = {self.K:_} (blocks)',
            xlabel='Blocks',
            ylabel='Improvement',
            filename=f'{PLOTS_PATH}improvement_{self.K}',
            color='black'
        )

