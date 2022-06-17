import warnings
import pandas as pd
from heuristic_analyzer import HeuristicAnalyzer

warnings.simplefilter(action='ignore', category=FutureWarning)

# Validated data-set files
VALIDATED_TXS_FILE = r'bitcoin-deanonymization/data/validated_transactions.csv'
VALIDATED_INPUTS_FILE = r'bitcoin-deanonymization/data/validated_inputs.csv'
VALIDATED_OUTPUTS_FILE = r'bitcoin-deanonymization/data/validated_outputs.csv'


def load_data(txf: str, inf: str, outf: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a .csv data-set into a pandas.Dataframe.

    :param <txf>: transactions file
    :param <inf>: inputs file
    :param <outf>: outputs file

    :return: three pandas dataframes (transactions, inputs, outputs)
    """

    # Load Transactions
    tx_data = pd.read_csv(txf)
    tx_df = pd.DataFrame(tx_data)
    tx_df.drop_duplicates(subset='id', keep='last', inplace=True)

    # Load Inputs
    in_data = pd.read_csv(inf)
    in_df = pd.DataFrame(in_data)

    # Load Outputs
    out_data = pd.read_csv(outf)
    out_df = pd.DataFrame(out_data)

    return tx_df, in_df, out_df   


def main() -> None:
    """Load dataset into pandas dataframes and apply the heuristics."""

    # Load (validated) dataset
    tx_df, in_df, out_df = load_data(
        VALIDATED_TXS_FILE,
        VALIDATED_INPUTS_FILE,
        VALIDATED_OUTPUTS_FILE)

    # Rename <sig_id> and <pk_id> to <address> for simplicity
    in_df.rename(columns = {'sig_id': 'address'}, inplace=True)
    out_df.rename(columns = {'pk_id': 'address'}, inplace=True)

    # Drop non-needed columns
    in_df.drop(columns=['id', 'out_id'], inplace=True)
    out_df.drop(columns=['id', 'value'], inplace=True)

    # Compute heuristics

    # Time-step for trend analysis (n. blocks)
    # 1 block -> 10 minutes
    blocks_one_month = ((365 * 24 * 60) / 12) / 10 # 1 month
    #blocks_two_month = blocks_one_month * 2 # 2 month
    #blocks_four_month = blocks_one_month * 4 # 4 month

    ha = HeuristicAnalyzer(tx_df, in_df, out_df, K=int(blocks_one_month))
    #ha = HeuristicAnalyzer(tx_df, in_df, out_df, K=int(blocks_two_month))
    #ha = HeuristicAnalyzer(tx_df, in_df, out_df, K=int(blocks_four_month))
    ha.mih() # MIH (Multi Input Addresses Heuristic)
    ha.och() # OCH (One-time Change Address Heuristic)
    ha.mih_and_och() # MIH + OCH

if __name__ == '__main__':
    main()
