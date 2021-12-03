import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime


def interp_bundle(df, bundle_map):
    df = df.sort_values('bundle_hash') # sort df by hash
    unique_bundles = df['bundle_hash'].unique()
    interp_bundle_game = df['gamecategory'].values.copy()
    interp_bundle_subgame = df['subgamecategory'].values.copy()
    bundles_array = df['bundle_hash'].values
    for bundle in tqdm(unique_bundles):
        if not bundle in bundle_map:
            continue
        game, subgame = bundle_map[bundle]
        start_index = np.searchsorted(bundles_array, bundle, side='left')
        fin_index = np.searchsorted(bundles_array, bundle, side='right')
        assert start_index != fin_index
        start_index = max(start_index - 2, 0)
        fin_index += 2
        bundles_array_cropped = bundles_array[start_index:fin_index]
        mask = bundles_array_cropped == bundle
        interp_bundle_game[start_index:fin_index][mask] = game
        interp_bundle_subgame[start_index:fin_index][mask] = subgame
        
    df['interp_game'] = interp_bundle_game
    df['interp_subgame'] = interp_bundle_subgame
    return df


def make_bundle_map(full_df):
    full_valid_df = full_df[full_df['gamecategory'].values == full_df['gamecategory'].values]
    full_valid_df = full_valid_df.reset_index()
    bundle_map = {}
    bundle_dict = full_valid_df['bundle_hash'].to_dict()
    bundle_dict = dict(zip(bundle_dict.values(), bundle_dict.keys()))
    for bundle in bundle_dict:
        bundle_index = bundle_dict[bundle]
        game = full_valid_df['gamecategory'].values[bundle_index]
        subgame = full_valid_df['subgamecategory'].values[bundle_index]
        bundle_map[bundle] = [game, subgame]
    return bundle_map


def main():
    """
    This script fills undefined Game and Subgame cols from dataset items with the same bundle 
    """
    train_df_path = 'train.csv'
    test_df_path = 'test.csv'
    print(datetime.datetime.now(), "Loading dfs")
    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)
    test_df['Segment'] = np.nan
    print(datetime.datetime.now(), "Concating dfs")
    test_df['bundle_hash'] = pd.util.hash_array(test_df["bundle"].values.copy())
    train_df['bundle_hash'] = pd.util.hash_array(train_df["bundle"].values.copy())

    full_df = pd.concat((train_df, test_df), sort=False)
    print(datetime.datetime.now(), "Making map")
    bundle_map = make_bundle_map(full_df)
    print(datetime.datetime.now(), "Running interp")
    test_df_interp_bundle = interp_bundle(test_df, bundle_map)
    test_df_interp_bundle.to_csv("test_df_interp_bundle.csv", index=None)
    train_df_interp_bundle = interp_bundle(train_df, bundle_map)
    train_df_interp_bundle.to_csv("train_df_interp_bundle.csv", index=None)


if __name__ == '__main__':
    main()