import numpy as np
import pandas as pd


def ver_to_number(ver):
    if '.' not in ver:
        return int(ver + '00')
    version_split = ver.split('.')
    if len(version_split) == 2:
        return int("".join(version_split) + '0')
    else:
        return int("".join(version_split))


def ios_versions_to_numerical(all_ios_versions):
    num_to_version = {}
    version_to_num = {}
    num_to_normalized_value = {}
    nums = []
    for version in all_ios_versions:
        if type(version) == float or version == 'iOS':
            continue
        version_split = version.split('.')
        if len(version_split) == 2:
            num = int("".join(version_split) + '0')
        else:
            num = int("".join(version_split))
        nums.append(num)
        version_to_num[version] = num

    for i, num in enumerate(sorted(nums)):
        num_to_normalized_value[num] = np.linspace(0, 1, len(nums))[i]

    for k in version_to_num.keys():
        version_to_num[k] = num_to_normalized_value[version_to_num[k]]

    version_to_num[np.nan] = np.nan
    version_to_num['iOS'] = np.nan
    return version_to_num


def android_versions_to_numerical(all_android_versions):
    def ver_to_number(ver):
        if '.' not in ver:
            return int(ver + '00')
        version_split = ver.split('.')
        if len(version_split) == 2:
            return int("".join(version_split) + '0')
        else:
            return int("".join(version_split))

    api_level_to_version = {
        '21': '5.0', '22': '5.1', '23': '6.0', '24': '7.0', '25': '7.1', '26': '8.0', '27': '8.1', '28': '9.0',
        '29': '10.0', '30': '11.0'
    }
    bad_key_to_num = {'7.0%20%5BSDK%2024%20%7C%20ARM%5D.0': 700, '8.1Go': 810, '6.0.99': 600, 'Android 7.0': 700,
                      '7.0_Lomaster_ROM.0': 700, '5.12.0': 590, '5.12': 590, }
    version_to_num = {}
    nums = []
    num_to_normalized_value = {}

    bad_keys = ['7.0%20%5BSDK%2024%20%7C%20ARM%5D.0', '8.1Go', '6.0.99', '7.0_Lomaster_ROM.0', '5.12.0', '5.12',
                'Android 7.0']
    parsed_sdks = []
    parsed_versions = []
    parsed_versions_keys = []

    nan_keys = [np.nan, '. . . ', '1000-7', '666.0.0', '0.42.0', '. . . ', '0.0.0']
    for version in all_android_versions:
        if version in nan_keys:
            continue
        if 'API' in version:
            parsed_version = version.split('/')[0][:-1]
            num = ver_to_number(parsed_version)
            nums.append(num)
            version_to_num[version] = num
        elif '(' in version:
            parsed_versions_keys.append(version)
            parsed_sdks.append(version.split('(')[0])
            parsed_sdk = version.split('(')[0]
            parsed_version = api_level_to_version[parsed_sdk]
            num = ver_to_number(parsed_version)
            nums.append(num)
            version_to_num[version] = num

        elif version in bad_keys:
            parsed_versions_keys.append(version)
            num = bad_key_to_num[version]
            nums.append(num)
            version_to_num[version] = num

        elif '.' in version:
            if int(version.split('.')[0]) <= 12:
                parsed_versions_keys.append(version)
                parsed_versions.append(version)
                num = ver_to_number(version)
                nums.append(num)
                version_to_num[version] = num
            else:
                parsed_versions_keys.append(version)
                nan_keys.append(version)
        else:
            if int(version) <= 12:
                parsed_versions.append(version)
                num = ver_to_number(version)
                nums.append(num)
                version_to_num[version] = num
                # process as android version
            elif int(version) >= 19:
                parsed_sdks.append(version)

                parsed_sdk = version.split('(')[0]
                parsed_version = api_level_to_version[parsed_sdk]
                num = ver_to_number(parsed_version)
                nums.append(num)
                version_to_num[version] = num
                # process as sdk version
            else:
                nan_keys.append(version)
    for i, num in enumerate(sorted(nums)):
        num_to_normalized_value[num] = np.linspace(0, 1, len(nums))[i]

    for k in version_to_num.keys():
        version_to_num[k] = num_to_normalized_value[version_to_num[k]]

    version_to_num[np.nan] = np.nan
    for nun_key in nan_keys:
        version_to_num[nun_key] = np.nan
    return version_to_num


def parse_version(df, and_v_to_num, ios_v_to_num):
    vers = []
    for os, v in zip(df['os'].values, df['osv'].values):
        if os == 'android':
            ver = and_v_to_num[v]
        elif os == 'ios':
            ver = ios_v_to_num[v]
        else:
            ver = np.nan
        vers.append(ver)
    df['osv_numerical'] = vers
    return df


def add_os_version_fetures(df):
    df[['os']] = df.os.str.lower()
    all_ios_versions = set(df[df.os.str.lower() == 'ios'].osv.unique())
    all_android_versions = set(df[df.os.str.lower() == 'android'].osv.unique())

    and_v_to_num = android_versions_to_numerical(all_android_versions)
    ios_v_to_num = ios_versions_to_numerical(all_ios_versions)
    df = parse_version(df, and_v_to_num, ios_v_to_num)
    return df


def get_timedelta(x):
    if x is None or (type(x) == float and np.isnan(x)):
        return x
    if x == 'MSK':
        return 0
    return int(x[3:])


def add_date_features(df):
    df['created'] = pd.to_datetime(df.created)
    df['created_local_tz'] = df['created'] + pd.to_timedelta(df['shift'].apply(get_timedelta), unit='hour')
    df['day_of_week'] = df['created_local_tz'].dt.weekday
    df['is_weekend'] = df['created_local_tz'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
    df['hour'] = df['created_local_tz'].dt.hour
    df['dist_from_msk_in_tz_hours'] = df['shift'].apply(get_timedelta).abs()
    return df


def processing_device(data):
    data = add_date_features(data)
    data = add_os_version_fetures(data)

    return data