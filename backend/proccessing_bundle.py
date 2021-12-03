import numpy as np
import pandas as pd


def interp_bundle(df, bundle_map):
    df = df.sort_values('bundle_hash')  # sort df by hash
    unique_bundles = df['bundle_hash'].unique()
    interp_bundle_game = df['gamecategory'].values.copy()
    interp_bundle_subgame = df['subgamecategory'].values.copy()
    bundles_array = df['bundle_hash'].values
    for bundle in unique_bundles:
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


def processing_bundle(data):
    if "Segment" in data.columns:
        if pd.isna(data["Segment"][0]):
            data['Segment'] = np.nan
    else:
        data['Segment'] = np.nan

    data['bundle_hash'] = pd.util.hash_array(data["bundle"].values.copy())

    bundle_map = make_bundle_map(data)
    data_interp_bundle = interp_bundle(data, bundle_map)
    return data_interp_bundle