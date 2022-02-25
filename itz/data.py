"""Data parsing for the American Consumer Survey and NYC PLUTO databases.
"""

import json
from tracemalloc import start
from typing import List, Tuple

import pandas as pd


ACS_DEMOGRAPHIC_PATH = "in-the-zone-data/acs/nyc-demographic-data-%s.csv"
ACS_ECONOMIC_PATH = "in-the-zone-data/acs/nyc-economic-data-%s.csv"
ACS_HOUSING_PATH = "in-the-zone-data/acs/nyc-housing-data-%s.csv"
ACS_SOCIAL_PATH = "in-the-zone-data/acs/nyc-social-data-%s.csv"
ACS_TRANSPORTATION_PATH = "in-the-zone-data/acs/nyc-transportation-data-%s.csv"
PLUTO_PATH = "in-the-zone-data/zoning-data/mergedPLUTO-%s.csv"
PLUTO_TEXT_PATH = "in-the-zone-data/zoning-data/mergedPLUTO-%s.txt"
TRACT_DICT_PATH = "in-the-zone-data/2000-to-2010-census-blocks-tracts.txt"
CENSUS_TRACT_GEODATA_PATH = "in-the-zone-data/ny_2010_census_tracts.json"
SUBSIDIZED_PROPERTIES_PATH = "in-the-zone-data/subsidized_properties.csv"
TRACTS_TO_LOTS_PATH = "in-the-zone-data/tracts-to-lots.json"

VAR_NAMES = ('Unnamed: 0', '2011_2016_percent_upzoned', '2011_2019_percent_upzoned',
       '2016_2019_percent_upzoned', '2011_2016_average_years_since_upzoning',
       '2011_2019_average_years_since_upzoning',
       '2016_2019_average_years_since_upzoning', 'd_2011_2016_resid_units',
       'd_2011_2019_resid_units', 'd_2016_2019_resid_units',
       'orig_percent_residential', 'orig_percent_limited_height',
       'orig_percent_mixed_development', 'orig_percent_subsidized_properties',
       'd_2011_2019_pop_density', 'd_2011_2019_resid_unit_density',
       'd_2011_2019_per_capita_income',
       'd_2011_2019_percent_non_hispanic_or_latino_white_alone',
       'd_2011_2019_percent_non_hispanic_black_alone',
       'd_2011_2019_percent_hispanic_any_race',
       'd_2011_2019_percent_non_hispanic_asian_alone',
       'd_2011_2019_percent_multi_family_units',
       'd_2011_2019_percent_occupied_housing_units',
       'd_2011_2019_median_gross_rent', 'd_2011_2019_median_home_value',
       'd_2011_2019_percent_households_with_people_under_18',
       'd_2011_2019_percent_of_households_in_same_house_year_ago',
       'd_2011_2019_percent_bachelor_degree_or_higher',
       'd_2011_2019_percent_car_commuters',
       'd_2011_2019_percent_public_transport_commuters',
       'd_2011_2019_mean_public_transport_travel_time',
       'd_2011_2019_mean_car_travel_time', 'orig_pop_density',
       'orig_percent_non_hispanic_or_latino_white_alone',
       'orig_percent_non_hispanic_black_alone',
       'orig_percent_hispanic_any_race',
       'orig_percent_non_hispanic_asian_alone', 'orig_median_age',
       'orig_per_capita_income', 'orig_resid_unit_density',
       'orig_percent_multi_family_units',
       'orig_percent_occupied_housing_units', 'orig_median_gross_rent',
       'orig_median_home_value',
       'orig_percent_households_with_people_under_18',
       'orig_percent_of_households_in_same_house_year_ago',
       'orig_percent_bachelor_degree_or_higher', 'orig_percent_car_commuters',
       'orig_percent_public_transport_commuters',
       'orig_mean_public_transport_travel_time', 'orig_mean_car_travel_time')

# DELTAS = [(2011, 2019), (2011, 2016), (2016, 2019)]
DELTAS = [("2011", "2019")]

# TRACT_DATA_YEARS = ["2011", "2016", "2019"]
TRACT_DATA_YEARS = ["2011", "2019"]
# LOT_DATA_YEARS = [str(year) for year in range(2011, 2020)]
LOT_DATA_YEARS = ["2011", "2019"]

COUNTY_CODES = {
    "005": "BX",
    "047": "BK",
    "061": "MN",
    "081": "QN",
    "085": "SI"
}
COUNTY_TO_CODE = {
    "BX": "005",
    "BK": "047",
    "MN": "061",
    "QN": "081",
    "SI": "085"
}

def get_data(lot_data: pd.DataFrame=None, tract_data: List[pd.DataFrame]=[],
             verbose=False) -> Tuple[pd.DataFrame, List[pd.DataFrame], pd.DataFrame]:
    """Creates DataFrame with columns corresponding to variables used in the SEM models.

    Parameters
    ----------
    lot_data (optional): pd.DataFrame
        Pre-parsed DataFrame containing lot-related data.
    tract_data (optional): List of pd.DataFrame
        Pre-parsed DataFrame containing tract-related data.
    verbose (optional): bool
        Whether to print status as the function executes.

    Returns
    -------
    Tuple
        Lot data DF, tract data DFs, and a combined DF for use with semopy models.
    """
    # Load/parse tract data.
    if verbose:
        print("Collecting tract data... ", end="")
    tract_dfs = []
    if len(tract_data) == 0:
        tract_dfs = _get_tract_data()
        if verbose:
            print("Done!")
    else:
        tract_dfs = tract_data
        if verbose:
            print("Using provided.")

    # Load lot data. 
    if verbose:
        print("Collecting lot data... ", end="")
    if lot_data is None:
        lot_df = _get_lot_data()
        if verbose:
            print("Done!")
    else:
        lot_df = lot_data
        if verbose:
            print("Using provided.")

    # Create dictionary which holds all lot BBL numbers corresponding to each tract ITZ_GEOID. 
    # tracts_to_lots = {}
    # for value in tract_dfs[0].index:
    #     tracts_to_lots[value] = []
    # for index, row in lot_df.iterrows():
    #     tracts_to_lots[row["ITZ_GEOID"]].append(index)
    # print("Tracts to lots created!")
    # with open("tract_to_lot_list.txt", "w") as f:
    #     f.write(str(tracts_to_lots))
    with open(TRACTS_TO_LOTS_PATH, "r") as f:
        tracts_to_lots = json.load(f)

    # Delete tracts without lots.  
    tracts_to_delete = []
    for tract, lot_list in tracts_to_lots.items():
        if len(lot_list) == 0:
            tracts_to_delete.append(tract)
    for tract in tracts_to_delete:
        del tracts_to_lots[tract]

    # Combine tract and lot data.
    if verbose:
        print("Producing lot-based data for tracts... ", end="") 
    tract_lot_data = _get_tract_lot_data(lot_df, tracts_to_lots)
    if verbose:
        print("Done!")

    # Find Tract delta data. 
    if verbose:
        print("Calculating deltas between starting and ending tract data... ", end="")
    tract_deltas = _get_delta_data(tract_dfs, tracts_to_lots.keys())
    if verbose:
        print("Done!")

    # Combine dataframes to create final model data. 
    if verbose:
        print("Combining data sources... ", end="")
    columns = {}
    for column in tract_dfs[0].columns:
        columns[column] = "orig_" + column
    tract_dfs[0].rename(mapper=columns, axis="columns", inplace=True)
    model_df = pd.concat([tract_lot_data, tract_deltas, tract_dfs[0]], axis=1)
    if verbose:
        print("Done!")
    return lot_df, tract_dfs, model_df


# TODO: Add verbosity options to these functions.


def _get_tract_data() -> List[pd.DataFrame]:
    """Returns a list of two DataFrames with columns not requiring lot data.
    Index: ITZ_GEOID
    """
    
    tract_dfs = []

    # Get tract area data.
    with open(CENSUS_TRACT_GEODATA_PATH, "r") as f:
        geodata = json.load(f)
    tract_area_df = pd.DataFrame(columns=["area"])
    tract_area_df.sort_index(inplace=True)
    # Add data row-by-row.
    for tract in geodata["features"]:
        if tract["properties"]["COUNTYFP10"] in COUNTY_CODES.keys():
            tract_area_df.loc[COUNTY_CODES[tract["properties"]["COUNTYFP10"]] + tract["properties"]["NAME10"]] = \
             tract["properties"]["ALAND10"]
    print(tract_area_df)
    # tract_area_df.sort_index(0)
    print("Tract area data collected")

    for year in TRACT_DATA_YEARS:
        tract_df = pd.DataFrame(index=tract_area_df.index)
        tract_df.index.rename('ITZ_GEOID', inplace=True)

        demographic = pd.read_csv(ACS_DEMOGRAPHIC_PATH % year, skiprows=[1], na_values=["(X)", "-", "**"])
        _add_tract_ids(demographic)
        demographic.set_index("ITZ_GEOID", inplace=True)
        demographic.sort_index(inplace=True)
        print(demographic)
        print(demographic.columns)
        tract_df["pop_density"] = demographic["DP05_0001E"].astype(float)
        tract_df["percent_non_hispanic_or_latino_white_alone"] = demographic["DP05_0072PE"].astype(float)
        tract_df["percent_non_hispanic_black_alone"] = demographic["DP05_0073PE"].astype(float)
        tract_df["percent_hispanic_any_race"] = demographic["DP05_0066PE"].astype(float)
        tract_df["percent_non_hispanic_asian_alone"] = demographic["DP05_0075PE"].astype(float)
        tract_df["median_age"] = demographic["DP05_0017E"].astype(float)
        del demographic
        print(tract_df)
        print(tract_df["pop_density"]["SI20.01"])
        print("Tract demographic data collected")

        economic = pd.read_csv(ACS_ECONOMIC_PATH % year, skiprows=[1], na_values=["(X)", "-", "**", "N"])
        _add_tract_ids(economic)
        economic.set_index("ITZ_GEOID", inplace=True)
        economic.sort_index(inplace=True)
        # Some tracts randomly have their value set to 'N' even though that's not reflected in the data
        tract_df["per_capita_income"] = economic["DP03_0088E"].astype(float)
        del economic

        print("Tract economic data collected")

        housing = pd.read_csv(ACS_HOUSING_PATH % year, skiprows=[1], na_values=["(X)", "-", "**", "2,000+", "1,000,000+"])
        _add_tract_ids(housing)
        housing.set_index("ITZ_GEOID", inplace=True)
        housing.sort_index(inplace=True)
        tract_df["resid_unit_density"] = housing["DP04_0001E"].astype(float)
        tract_df["percent_multi_family_units"] = 100 - housing["DP04_0007PE"].astype(float) - housing["DP04_0008PE"].astype(float)
        tract_df["percent_occupied_housing_units"] = housing["DP04_0002PE"].astype(float)
        # 86 tracts have gross rent = '2000+'
        # print(housing["GEO_ID"][housing["DP04_0132E"] == "2,000+"])
        tract_df["median_gross_rent"] = housing["DP04_0132E"].astype(float)
        # 110 tracts have median house value = '1,000,000+'
        # print(housing["GEO_ID"][housing["DP04_0088E"] == "1,000,000+"])
        tract_df["median_home_value"] = housing["DP04_0088E"].astype(float)
        del housing

        print("Tract housing data collected")

        social = pd.read_csv(ACS_SOCIAL_PATH % year, skiprows=[1], na_values=["(X)", "-", "**"])
        _add_tract_ids(social)
        social.set_index("ITZ_GEOID", inplace=True)
        social.sort_index(inplace=True)
        tract_df["percent_households_with_people_under_18"] = social["DP02_0013PE"].astype(float)
        tract_df["percent_of_households_in_same_house_year_ago"] = social["DP02_0079PE"].astype(float)
        tract_df["percent_bachelor_degree_or_higher"] = social["DP02_0067PE"].astype(float)
        del social

        print("Tract social data collected")

        transportation = pd.read_csv(ACS_TRANSPORTATION_PATH % year, skiprows=[1], na_values=["(X)", "-", "**", "N"])
        _add_tract_ids(transportation)
        transportation.set_index("ITZ_GEOID", inplace=True)
        transportation.sort_index(inplace=True)
        tract_df["percent_car_commuters"] = \
            (transportation["S0802_C02_001E"].astype(float) + transportation["S0802_C03_001E"].astype(float)) \
          / transportation["S0802_C01_001E"].astype(float)
        tract_df["percent_public_transport_commuters"] = transportation["S0802_C04_001E"] \
                                                       / transportation["S0802_C01_001E"]
        # TODO: find a fix for the fact that 1800 blocks have no value for travel times, and percentages are highly unreliable
        # print(transportation["GEO_ID"][transportation["S0802_C01_090E"] == "N"])
        tract_df["mean_public_transport_travel_time"] = transportation["S0802_C04_090E"].astype(float)
        tract_df["mean_car_travel_time"] = transportation["S0802_C03_090E"].astype(float)
        # _add_tract_ids(transportation)
        # tract_df["ITZ_GEOID"] = transportation["ITZ_GEOID"]

        # tract_df.set_index("ITZ_GEOID", inplace=True)

        print(tract_df)
        print(tract_df.index)
        # Divide all density columns by tract area. 
        for tract_id in tract_area_df.index:
            tract_area = tract_area_df.at[tract_id, "area"]
            tract_df["pop_density"][tract_id] /= tract_area
            # tract_df["pop_density"][tract_df.loc[tract_df["ITZ_GEOID"] == tract_id]] /= tract_area
            tract_df["resid_unit_density"][tract_id] /= tract_area
            # tract_df["resid_unit_density"][tract_df.loc[tract_df["ITZ_GEOID"] == tract_id]] /= tract_area

        del transportation

        print("Tract transportation data collected")

        tract_dfs.append(tract_df)

    return tract_dfs

def _add_tract_ids(tract_df: pd.DataFrame):
    """Adds an "ITZ_GEOID" column to the data combining the borough and census tract number.
    """
    try:
        # print(tract_df["GEOID10"].str.slice(start=11, stop=14))
        itz_geoids = []
        for _, row in tract_df.iterrows():
            geoid = row["GEOID10"]
            if geoid[-1] != '0':
                itz_geoids.append(COUNTY_CODES[geoid[11:14]] + row["NAME"][13:row["NAME"].find(",")])
            else:
                itz_geoids.append(COUNTY_CODES[geoid[11:14]] + row["NAME"][13:row["NAME"].find(",")])
    except:
        # print(tract_df["GEO_ID"].str.slice(start=11, stop=14))
        itz_geoids = []
        for _, row in tract_df.iterrows():
            geoid = row["GEO_ID"]
            if geoid[-1] != '0':
                itz_geoids.append(COUNTY_CODES[geoid[11:14]] + row["NAME"][13:row["NAME"].find(",")])
            else:
                itz_geoids.append(COUNTY_CODES[geoid[11:14]] + row["NAME"][13:row["NAME"].find(",")])

    tract_df["ITZ_GEOID"] = itz_geoids


def _get_lot_data() -> pd.DataFrame:
    """Creates DataFrame with columns being lot-specific data and rows being lots. 
    Index: ITZ_GEOID
    Uses 2011 Lots - all other lots are discarded. 
    """
    try:
        starting_pluto = pd.read_csv(PLUTO_PATH % LOT_DATA_YEARS[0], header=0, dtype=str)
    except:
        starting_pluto = pd.read_table(PLUTO_TEXT_PATH % LOT_DATA_YEARS[0], header=0, sep=",", dtype=str, usecols=["BBL", "Version", "Borough", "LotArea"])
        next_pluto = pd.read_table(PLUTO_TEXT_PATH % 2012, header=0, sep=",", dtype=str, usecols=["BBL", "CT2010"])
        print("next pluto created")
        next_pluto.set_index("BBL", inplace=True)
    print("starting pluto created")
    # print(starting_pluto.columns)
    # print(starting_pluto.index)
    starting_pluto.set_index("BBL", inplace=True)
    starting_pluto.sort_index()
    print(starting_pluto)

    # Create ITZ_GEOID column in lot_df
    if starting_pluto["Version"].iloc[0] == "11v1  ":
        # next_pluto = pd.read_table(PLUTO_TEXT_PATH % 2012, header=0, sep=",", dtype=str, usecols=["BBL", "CT2010"])
        print(next_pluto)
        # itz_geoid = []
        # success = 0
        starting_pluto = starting_pluto.join(next_pluto, on="BBL")
        del next_pluto
        print(starting_pluto)
        print(starting_pluto.columns)
        starting_pluto["ITZ_GEOID"] = starting_pluto["Borough"] + starting_pluto["CT2010"].str.strip()
        # starting_pluto["ITZ_GEOID"].mask[starting_pluto["ITZ_GEOID"] in ["MN","QN","BX","SI","BK"]]
        # for index, row in starting_pluto.iterrows():
        #     try:
        #         tract_2010 = next_pluto.loc[next_pluto["BBL"] == index]["CT2010"].strip()
        #         county = COUNTY_CODES[row["Borough"]]
        #         itz_geoid.append(county+tract_2010)
        #         success += 1
        #     except:
        #         itz_geoid.append("nan")

        # block_to_tract = {}
        # tract_to_tract = {}
        # census_conversion_df = pd.read_csv(TRACT_DICT_PATH, dtype=str)
        # # print(census_conversion_df[census_conversion_df["BLK_2000"] == ''])
        # for _, row in census_conversion_df.iterrows():
        #     if row["TRACT_2010"][-1] != "0":
        #         # 0850176003015
        #         # 0850208001037
        #         # 0610208003001
        #         block_to_tract[row["COUNTY_2000"] + row["TRACT_2000"] + row["BLK_2000"]] = str(int(row["TRACT_2010"][:-2])) + "." + str(int(row["TRACT_2010"][-2:]))
        #         tract_to_tract[row["COUNTY_2000"] + row["TRACT_2000"]] = str(int(row["TRACT_2010"][:-2])) + "." + str(int(row["TRACT_2010"][-2:]))
        #     else:
        #         block_to_tract[row["COUNTY_2000"] + row["TRACT_2000"] + row["BLK_2000"]] = str(int(row["TRACT_2010"][:-2]))
        #         tract_to_tract[row["COUNTY_2000"] + row["TRACT_2000"]] = str(int(row["TRACT_2010"][:-2]))
        # census_2000 = starting_pluto["Borough"].map(COUNTY_TO_CODE) + starting_pluto["Tract2000"].astype(str).str.slice(stop=4)  +  "00" + starting_pluto["CB2000"].astype(str).str.slice(stop=4)
        # census_2000.to_csv("pluto census 2000.csv")
        # with open("block-to-tract.csv", "w") as f:
        #     f.write(str(block_to_tract))
        # itz_geoid = []
        # num_good = 0
        # for index, id in census_2000.items():
        #     try:
        #         itz_geoid.append(starting_pluto.at[index, "Borough"] + block_to_tract[id])
        #     except:
        #         itz_geoid.append("Nan")
        #         # try: 
        #         #     itz_geoid.append(starting_pluto.at[index, "Borough"] + tract_to_tract[id])
        #         # except:
        #         #     print(id)
        #         # print(id)
        #         continue
        #     else:
        #         print("OK AND CORRECT!!!", id)
        #         num_good += 1
        #         # print(starting_pluto.loc[index], starting_pluto.at[index, "Tract2000"], starting_pluto.at[index, "CB2000"])
        # starting_pluto["ITZ_GEOID"] = pd.Series(itz_geoid)
    else: 
        starting_pluto["ITZ_GEOID"] = starting_pluto["Borough"] + starting_pluto["CT2010"]

    # print(num_good)
    # print(success)
    starting_pluto = starting_pluto[starting_pluto['ITZ_GEOID'].map(lambda x: len(str(x)) != 2)]
    starting_pluto = starting_pluto[starting_pluto["ITZ_GEOID"].notnull()]
    lot_bbl = starting_pluto.index
    # print(len(itz_geoid))
    columns = []
    for year in LOT_DATA_YEARS:
        columns.append("land_use"+year)
        columns.append("zoning"+year)
        columns.append("max_resid_far"+year)
        columns.append("mixed_development"+year)
        columns.append("limited_height"+year)
        columns.append("resid_units"+year)
    lot_df = pd.DataFrame(index=lot_bbl, columns=columns)
    print(len(lot_df))
    lot_df["ITZ_GEOID"] = starting_pluto["ITZ_GEOID"]
    print(lot_df["ITZ_GEOID"])
    # LotArea doesn't change, and starting_pluto uses the same indexing as lot_df, so the column can simply be copied over.
    lot_df["lot_area"] = starting_pluto["LotArea"].astype(float)
    print("ITZ geoids created!")

    del starting_pluto

    for year in LOT_DATA_YEARS:
        print("Beginning: ", year)
        try:
            pluto_df = pd.read_csv(PLUTO_PATH % year, dtype=str)
        except:
            pluto_df = pd.read_csv(PLUTO_TEXT_PATH % year, sep=",", dtype=str)
        # Sort by BBL
        print(pluto_df)
        try:
            pluto_df.set_index('BBL', inplace=True)
            bbl_var = 'BBL'
        except:
            pluto_df.set_index('bbl', inplace=True)
            bbl_var = 'bbl'
        pluto_df.sort_index()
        pluto_df = pluto_df.loc[lot_df.index.intersection(pluto_df.index)]
        # pluto_df = pluto_df.loc[lot_df.index]
        print(pluto_df)
        try:
            lot_df["land_use" + year] = pluto_df["LandUse"]
        except:
            lot_df["land_use" + year] = pluto_df["landuse"]
        # except:
        #     print("EXCEPTED AT LINE 281")
        #     continue
        try:
            # lot_df["zoning" + year] = pluto_df["ZoneDist1"][pluto_df[bbl_var].searchsorted(bbl)]
            lot_df["zoning" + year] = pluto_df["ZoneDist1"]
        except:
            # lot_df["zoning" + year] = pluto_df["zonedist1"][pluto_df[bbl_var].searchsorted(bbl)]
            lot_df["zoning" + year] = pluto_df["zonedist1"]

        try:
            # lot_df["max_resid_far" + year] = pluto_df["ResidFAR"][pluto_df[bbl_var].searchsorted(bbl)].astype(float)
            lot_df["max_resid_far" + year] = pluto_df["ResidFAR"]
        except: 
            # lot_df["max_resid_far" + year] = pluto_df["MaxAllwFAR"][pluto_df[bbl_var].searchsorted(bbl)].astype(float)
            try:
                lot_df["max_resid_far" + year] = pluto_df["MaxAllwFAR"]
            except:
                lot_df["max_resid_far" + year] = pluto_df["residfar"]
        # for bbl in lot_bbl:
        #     print(bbl)
        #     # try: 
        #         # Find land use of lot with this BBL using binary search
        #     # lot_df["land_use" + year][bbl] = pluto_df["LandUse"][pluto_df[bbl_var].searchsorted(bbl)]
        #     # lot_df["Zoning" + year][bbl] = pluto_df.at[bbl, "LandUse"]
        #     # print(pluto_df["LandUse"])
        #     # print(pluto_df.index)
        #     # print(pluto_df.loc[[bbl]])
        #     # pluto_df["LandUse"].to_csv("landuse.csv")
        #     # print(pluto_df.at[bbl, "LandUse"])
        #     try:
        #         lot_df["land_use" + year][bbl] = pluto_df["LandUse"][bbl]
        #     except:
        #         lot_df["land_use" + year][bbl] = pluto_df["landuse"][bbl]
        #     # except:
        #     #     print("EXCEPTED AT LINE 281")
        #     #     continue
        #     try:
        #         lot_df["zoning" + year][bbl] = pluto_df["ZoneDist1"][pluto_df[bbl_var].searchsorted(bbl)]
        #         lot_df["zoning" + year][bbl] = pluto_df["ZoneDist1"][bbl]
        #     except:
        #         lot_df["zoning" + year][bbl] = pluto_df["zonedist1"][pluto_df[bbl_var].searchsorted(bbl)]
        #         lot_df["zoning" + year][bbl] = pluto_df["zonedist1"][bbl]

        #     try:
        #         lot_df["max_resid_far" + year][bbl] = pluto_df["ResidFAR"][pluto_df[bbl_var].searchsorted(bbl)].astype(float)
        #         lot_df["max_resid_far" + year][bbl] = pluto_df["ResidFAR"][bbl]
        #     except: 
        #         # lot_df["max_resid_far" + year][bbl] = pluto_df["MaxAllwFAR"][pluto_df[bbl_var].searchsorted(bbl)].astype(float)
        #         try:
        #             lot_df["max_resid_far" + year][bbl] = pluto_df["MaxAllwFAR"][bbl]
        #         except:
        #             lot_df["max_resid_far" + year][bbl] = pluto_df["residfar"][bbl]

        try:
            lot_df["mixed_development" + year] = pluto_df["landuse"] == "4"
        except KeyError:
            lot_df["mixed_development" + year] = pluto_df["LandUse"] == "4"
        try:
            lot_df["limited_height" + year] = pluto_df["ltdheight"].str.strip() != ''
        except KeyError:
            lot_df["limited_height" + year] = pluto_df["LtdHeight"].str.strip() != ''
        try:
            lot_df["resid_units" + year] = pluto_df["unitsres"]
        except KeyError:
            lot_df["resid_units" + year] = pluto_df["UnitsRes"]

        del pluto_df
    print("Lot Data year data collected!")
        # Boolean series
    # lot_df.reset_index(inplace=True, drop=True)
    # lot_df.set_index("ITZ_GEOID", inplace=True)
    # lot_df.sort_index()
    return lot_df


def _get_tract_lot_data(lot_df, tracts_to_lots) -> pd.DataFrame:
    """Calculates the percent of each tract that was upzoned using the (using a 10% threshold in
    maximum residential capacity)
    """
    tract_lot_data = pd.DataFrame(index=tracts_to_lots.keys(), columns=[
        "2011_2016_percent_upzoned", 
        "2011_2019_percent_upzoned", 
        "2016_2019_percent_upzoned", 
        "2011_2016_average_years_since_upzoning", 
        "2011_2019_average_years_since_upzoning", 
        "2016_2019_average_years_since_upzoning", 
        "d_2011_2016_resid_units",
        "d_2011_2019_resid_units",
        "d_2016_2019_resid_units", 
        "orig_percent_residential",
        "orig_percent_limited_height",
        "orig_percent_mixed_development",
        "orig_percent_subsidized_properties"
    ])
    for delta in DELTAS:
        start = delta[0]
        end = delta[1]
        upzoned = 0
        years_since_upzoned = []
        residential_units_start = 0
        residential_units_end = 0
        for tract, lot_list in tracts_to_lots.items():
            for lot in lot_list:
                prev = float(lot_df.at[lot, "max_resid_far"+str(start)])*lot_df.at[lot, "lot_area"]
                # for year in range(start+1, end+1):
                for year in [2011, 2019]:
                    curr = float(lot_df.at[lot, "max_resid_far"+str(year)])*lot_df.at[lot, "lot_area"]
                    if prev != 0 and curr/prev > 1.1:
                        upzoned += 1
                        years_since_upzoned.append(int(end)-int(year))
                        break
                    prev = curr
                try:
                    residential_units_start += float(lot_df.at[lot, "resid_units"+str(start)])
                    residential_units_end += float(lot_df.at[lot, "resid_units"+str(end)])
                except ValueError:
                    continue
            try:
                tract_lot_data.at[tract, start + "_" + end + "_percent_upzoned"] = upzoned/len(lot_list)
                tract_lot_data.at[tract, start + "_" + end + "_average_years_since_upzoning"] = sum(years_since_upzoned)/len(years_since_upzoned)
                tract_lot_data.at[tract, "d_" + start + "_" + end + "_resid_units"] = residential_units_end-residential_units_start
            except:
                print(tract, len(lot_list), lot_list)
    land_use_lots = 0
    for tract, lot_list in tracts_to_lots.items():
        residential = 0
        limited_height = 0
        mixed_development = 0
        for lot in lot_list:    
            # 1 corresponds to one/two family, 2 and 3 correspond to multi-family, 4 corresponds to mixed_resid/comm
            try:
                if int(lot_df["land_use2011"][lot]) < 5:
                    residential += 1
                land_use_lots += 1
            except ValueError:
                pass
            if lot_df["limited_height2011"][lot] == True:
                limited_height += 1
            if lot_df["mixed_development2011"][lot] == True:
                mixed_development += 1
        tract_lot_data.at[tract, "orig_percent_residential"] = residential/land_use_lots
        tract_lot_data.at[tract, "orig_percent_mixed_development"] = mixed_development/land_use_lots
        tract_lot_data.at[tract, "orig_percent_limited_height"] = limited_height/len(lot_list)

    subsidized_property_by_tract = pd.read_csv(SUBSIDIZED_PROPERTIES_PATH)["tract_10"].value_counts(sort=False)
    for tract, count in subsidized_property_by_tract.items():
        tract_lot_data.at[tract, "orig_percent_subsidized_properties"] = count/len(lot_list)
    return tract_lot_data


def _get_delta_data(tract_dfs, index) -> pd.DataFrame:
    """Calculates the changes for tract-specific data between starting and ending points. 
    """
    tract_delta = pd.DataFrame(index=index)
    # date_to_index = {'2011':0, '2016':1, '2019':2}
    date_to_index = {'2011':0, '2019':1}

    for delta in DELTAS:
        start_index = date_to_index[delta[0]]
        end_index = date_to_index[delta[1]]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_pop_density"] = tract_dfs[end_index]["pop_density"] - tract_dfs[start_index]["pop_density"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_resid_unit_density"] = tract_dfs[end_index]["resid_unit_density"] - tract_dfs[start_index]["resid_unit_density"]
        
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_per_capita_income"] = tract_dfs[end_index]["per_capita_income"] - tract_dfs[start_index]["per_capita_income"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_percent_non_hispanic_or_latino_white_alone"] = tract_dfs[end_index]["percent_non_hispanic_or_latino_white_alone"] - tract_dfs[start_index]["percent_non_hispanic_or_latino_white_alone"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_percent_non_hispanic_black_alone"] = tract_dfs[end_index]["percent_non_hispanic_black_alone"] - tract_dfs[start_index]["percent_non_hispanic_black_alone"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_percent_hispanic_any_race"] = tract_dfs[end_index]["percent_hispanic_any_race"] - tract_dfs[start_index]["percent_hispanic_any_race"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_percent_non_hispanic_asian_alone"] = tract_dfs[end_index]["percent_non_hispanic_asian_alone"] - tract_dfs[start_index]["percent_non_hispanic_asian_alone"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_percent_multi_family_units"] = tract_dfs[end_index]["percent_multi_family_units"] - tract_dfs[start_index]["percent_multi_family_units"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_percent_occupied_housing_units"] = tract_dfs[end_index]["percent_occupied_housing_units"] - tract_dfs[start_index]["percent_occupied_housing_units"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_median_gross_rent"] = tract_dfs[end_index]["median_gross_rent"] - tract_dfs[start_index]["median_gross_rent"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_median_home_value"] = tract_dfs[end_index]["median_home_value"] - tract_dfs[start_index]["median_home_value"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_percent_households_with_people_under_18"] = tract_dfs[end_index]["percent_households_with_people_under_18"] - tract_dfs[start_index]["percent_households_with_people_under_18"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_percent_of_households_in_same_house_year_ago"] = tract_dfs[end_index]["percent_of_households_in_same_house_year_ago"] - tract_dfs[start_index]["percent_of_households_in_same_house_year_ago"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_percent_bachelor_degree_or_higher"] = tract_dfs[end_index]["percent_bachelor_degree_or_higher"] - tract_dfs[start_index]["percent_bachelor_degree_or_higher"]
        
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_percent_car_commuters"] = tract_dfs[end_index]["percent_car_commuters"] - tract_dfs[start_index]["percent_car_commuters"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_percent_public_transport_commuters"] = tract_dfs[end_index]["percent_public_transport_commuters"] - tract_dfs[start_index]["percent_public_transport_commuters"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_mean_public_transport_travel_time"] = tract_dfs[end_index]["mean_public_transport_travel_time"] - tract_dfs[start_index]["mean_public_transport_travel_time"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_mean_car_travel_time"] = tract_dfs[end_index]["mean_car_travel_time"] - tract_dfs[start_index]["mean_car_travel_time"]
        
    return tract_delta