"""Data parsing for the American Consumer Survey and NYC PLUTO databases.
"""

import json
from tracemalloc import start
from typing import List, Tuple

import pandas as pd
import numpy as np
import math


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

VAR_NAMES = ('all_vars', '2011_2016_percent_upzoned', '2011_2019_percent_upzoned',
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
       'd_2011_2019_percent_public_transport_trips_under_45_min',
       'd_2011_2019_percent_car_trips_under_45_min', 'orig_pop_density',
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
       'orig_percent_public_transport_trips_under_45_min', 'orig_percent_car_trips_under_45_min')
       
# VAR_NAMES = ('2011_2016_percent_upzoned', '2011_2019_percent_upzoned', '2016_2019_percent_upzoned', '2011_2016_average_years_since_upzoning', '2011_2019_average_years_since_upzoning', '2016_2019_average_years_since_upzoning', 'd_2011_2016_resid_units', 'd_2011_2019_resid_units', 'd_2016_2019_resid_units', 'orig_percent_residential', 'orig_percent_limited_height', 'orig_percent_mixed_development', 'orig_percent_subsidized_properties', 'd_2011_2019_pop_density', 'd_2011_2019_resid_unit_density', 'd_2011_2019_per_capita_income', 'd_2011_2019_percent_non_hispanic_or_latino_white_alone', 'd_2011_2019_percent_non_hispanic_black_alone', 'd_2011_2019_percent_hispanic_any_race', 'd_2011_2019_percent_non_hispanic_asian_alone', 'd_2011_2019_percent_multi_family_units', 'd_2011_2019_percent_occupied_housing_units', 'd_2011_2019_median_gross_rent', 'd_2011_2019_median_home_value', 'd_2011_2019_percent_households_with_people_under_18', 'd_2011_2019_percent_of_households_in_same_house_year_ago', 'd_2011_2019_percent_bachelor_degree_or_higher', 'd_2011_2019_percent_car_commuters', 'd_2011_2019_percent_public_transport_commuters', 'd_2011_2019_percent_public_transport_trips_under_45_min', 'd_2011_2019_percent_car_trips_under_45_min', 'd_2011_2016_pop_density', 'd_2011_2016_resid_unit_density', 'd_2011_2016_per_capita_income', 'd_2011_2016_percent_non_hispanic_or_latino_white_alone', 'd_2011_2016_percent_non_hispanic_black_alone', 'd_2011_2016_percent_hispanic_any_race', 'd_2011_2016_percent_non_hispanic_asian_alone', 'd_2011_2016_percent_multi_family_units', 'd_2011_2016_percent_occupied_housing_units', 'd_2011_2016_median_gross_rent', 'd_2011_2016_median_home_value', 'd_2011_2016_percent_households_with_people_under_18', 'd_2011_2016_percent_of_households_in_same_house_year_ago', 'd_2011_2016_percent_bachelor_degree_or_higher', 'd_2011_2016_percent_car_commuters', 'd_2011_2016_percent_public_transport_commuters', 'd_2011_2016_percent_public_transport_trips_under_45_min', 'd_2011_2016_percent_car_trips_under_45_min', 'd_2016_2019_pop_density', 'd_2016_2019_resid_unit_density', 'd_2016_2019_per_capita_income', 'd_2016_2019_percent_non_hispanic_or_latino_white_alone', 'd_2016_2019_percent_non_hispanic_black_alone', 'd_2016_2019_percent_hispanic_any_race', 'd_2016_2019_percent_non_hispanic_asian_alone', 'd_2016_2019_percent_multi_family_units', 'd_2016_2019_percent_occupied_housing_units', 'd_2016_2019_median_gross_rent', 'd_2016_2019_median_home_value', 'd_2016_2019_percent_households_with_people_under_18', 'd_2016_2019_percent_of_households_in_same_house_year_ago', 'd_2016_2019_percent_bachelor_degree_or_higher', 'd_2016_2019_percent_car_commuters', 'd_2016_2019_percent_public_transport_commuters', 'd_2016_2019_percent_public_transport_trips_under_45_min', 'd_2016_2019_percent_car_trips_under_45_min', 'orig_pop_density', 'orig_percent_non_hispanic_or_latino_white_alone', 'orig_percent_non_hispanic_black_alone', 'orig_percent_hispanic_any_race', 'orig_percent_non_hispanic_asian_alone', 'orig_median_age', 'orig_per_capita_income', 'orig_resid_unit_density', 'orig_percent_multi_family_units', 'orig_percent_occupied_housing_units', 'orig_median_gross_rent', 'orig_median_home_value', 'orig_percent_households_with_people_under_18', 'orig_percent_of_households_in_same_house_year_ago', 'orig_percent_bachelor_degree_or_higher', 'orig_percent_car_commuters', 'orig_percent_public_transport_commuters', 'orig_percent_public_transport_trips_under_45_min', 'orig_percent_car_trips_under_45_min')

DELTAS = [("2011", "2019"), ("2011", "2016"), ("2016", "2019")]

TRACT_DATA_YEARS = ["2011", "2016", "2019"]
LOT_DATA_YEARS = [str(year) for year in range(2011, 2020)]
# LOT_DATA_YEARS = ["2011", "2016", "2019"]

CODE_TO_COUNTY = {
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
SQFT_TO_SQKM = 10763910.41671


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

    for tract_df in tract_dfs:
        tract_df = tract_df[tract_df["median_gross_rent"].notnull()]
        tract_df = tract_df[tract_df["median_home_value"].notnull()]
    # TODO: GENERALIZE INDEX INTERSECTIONS
    tract_dfs[0] = tract_dfs[0].loc[tract_dfs[0].index.intersection(tract_dfs[1].index).intersection(tract_dfs[2].index)]
    tract_dfs[1] = tract_dfs[1].loc[tract_dfs[0].index.intersection(tract_dfs[1].index).intersection(tract_dfs[2].index)]
    tract_dfs[2] = tract_dfs[2].loc[tract_dfs[0].index.intersection(tract_dfs[1].index).intersection(tract_dfs[2].index)]

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
    try:
        lot_df.set_index("BBL", inplace=True)
    except:
        pass

    # Create dictionary which holds all lot BBL numbers corresponding to each tract ITZ_GEOID. 
    tracts_to_lots = {}
    for value in tract_dfs[0].index:
        tracts_to_lots[value] = []
    for index, row in lot_df.iterrows():
        # print(row)
        # print(lot_df.columns)
        tracts_to_lots[row["ITZ_GEOID"]].append(index)
    print("Tracts to lots created!")
    with open(TRACTS_TO_LOTS_PATH, "w") as f:
        json.dump(tracts_to_lots, f)
    # with open(TRACTS_TO_LOTS_PATH, "r") as f:
    #     tracts_to_lots = json.load(f)

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
    # model_df.to_csv("itz-data.csv")
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
        if tract["properties"]["COUNTYFP10"] in CODE_TO_COUNTY.keys():
            tract_area_df.loc[CODE_TO_COUNTY[tract["properties"]["COUNTYFP10"]] + tract["properties"]["NAME10"]] = \
             tract["properties"]["ALAND10"]

    print("Tract area data collected")

    for year in TRACT_DATA_YEARS:
        print(year, "TRACT_YEAR")
        # Create tract dataframe to store data in. 
        tract_df = pd.DataFrame(index=tract_area_df.index)
        tract_df.index.rename('ITZ_GEOID', inplace=True)

        # Load ACS demographic data. 
        demographic = pd.read_csv(ACS_DEMOGRAPHIC_PATH % year, skiprows=[1], na_values=["(X)", "-", "**"])
        # Create ITZ_GEOID column and sort it so it aligns with tract_df index. 
        _add_tract_ids(demographic)
        demographic.set_index("ITZ_GEOID", inplace=True)
        demographic.sort_index(inplace=True)
        # Load dictionary which maps ACS codes to columns
        with open("in-the-zone-data/acs/code-to-column-demographic-data-"+str(year)+".txt", "r") as f:
            code_to_column = eval(f.read())

        if year == "2011":
            tract_df["pop_density"] = demographic[code_to_column['Estimate!!SEX AND AGE!!Total population']].astype(float)
            tract_df["percent_non_hispanic_or_latino_white_alone"] = demographic[code_to_column['Percent!!RACE!!One race!!White']].astype(float)
            tract_df["percent_non_hispanic_black_alone"] = demographic[code_to_column['Percent!!RACE!!One race!!Black or African American']].astype(float)
            tract_df["percent_hispanic_any_race"] = demographic[code_to_column['Percent!!HISPANIC OR LATINO AND RACE!!Hispanic or Latino (of any race)']].astype(float)
            tract_df["percent_non_hispanic_asian_alone"] = demographic[code_to_column['Percent!!RACE!!One race!!Asian']].astype(float)
            tract_df["median_age"] = demographic[code_to_column['Estimate!!SEX AND AGE!!Median age (years)']].astype(float)
        elif year == "2019":
            tract_df["pop_density"] = demographic[code_to_column['Estimate!!SEX AND AGE!!Total population']].astype(float)
            tract_df["percent_non_hispanic_or_latino_white_alone"] = demographic[code_to_column['Percent!!HISPANIC OR LATINO AND RACE!!Total population!!Not Hispanic or Latino!!White alone']].astype(float)
            tract_df["percent_non_hispanic_black_alone"] = demographic[code_to_column['Percent!!HISPANIC OR LATINO AND RACE!!Total population!!Not Hispanic or Latino!!Black or African American alone']].astype(float)
            tract_df["percent_hispanic_any_race"] = demographic[code_to_column['Percent!!HISPANIC OR LATINO AND RACE!!Total population!!Hispanic or Latino (of any race)']].astype(float)
            tract_df["percent_non_hispanic_asian_alone"] = demographic[code_to_column['Percent!!HISPANIC OR LATINO AND RACE!!Total population!!Not Hispanic or Latino!!Asian alone']].astype(float)
            tract_df["median_age"] = demographic[code_to_column['Estimate!!SEX AND AGE!!Total population!!Median age (years)']].astype(float)
        else:
            tract_df["pop_density"] = demographic[code_to_column['Estimate!!SEX AND AGE!!Total population']].astype(float)
            tract_df["percent_non_hispanic_or_latino_white_alone"] = demographic[code_to_column['Percent!!HISPANIC OR LATINO AND RACE!!Total population!!Not Hispanic or Latino!!White alone']].astype(float)
            tract_df["percent_non_hispanic_black_alone"] = demographic[code_to_column['Percent!!HISPANIC OR LATINO AND RACE!!Total population!!Not Hispanic or Latino!!Black or African American alone']].astype(float)
            tract_df["percent_hispanic_any_race"] = demographic[code_to_column['Percent!!HISPANIC OR LATINO AND RACE!!Total population!!Hispanic or Latino (of any race)']].astype(float)
            tract_df["percent_non_hispanic_asian_alone"] = demographic[code_to_column['Percent!!HISPANIC OR LATINO AND RACE!!Total population!!Not Hispanic or Latino!!Asian alone']].astype(float)
            tract_df["median_age"] = demographic[code_to_column['Estimate!!SEX AND AGE!!Median age (years)']].astype(float)

        del demographic

        print("Tract demographic data collected")

        # Load ACS economic data.  
        # Some tracts randomly have their value for per_capita_income set to 'N' even though that's not reflected in the data
        economic = pd.read_csv(ACS_ECONOMIC_PATH % year, skiprows=[1], na_values=["(X)", "-", "**", "N"])
        # economic = pd.read_csv(ACS_ECONOMIC_PATH % year, skiprows=[1], na_values=["(X)", "-", "**"])
        # Create ITZ_GEOID column and sort it so it aligns with tract_df index. 
        _add_tract_ids(economic)
        economic.set_index("ITZ_GEOID", inplace=True)
        economic.sort_index(inplace=True)
        # Load dictionary which maps ACS codes to columns
        with open("in-the-zone-data/acs/code-to-column-economic-data-"+str(year)+".txt", "r") as f:
            code_to_column = eval(f.read())
        print(economic.loc[economic[code_to_column['Estimate!!INCOME AND BENEFITS (IN '+ year +' INFLATION-ADJUSTED DOLLARS)!!Per capita income (dollars)']] == "N"])
        tract_df["per_capita_income"] = economic[code_to_column['Estimate!!INCOME AND BENEFITS (IN '+ year +' INFLATION-ADJUSTED DOLLARS)!!Per capita income (dollars)']].astype(float)
        
        del economic

        print("Tract economic data collected")

        # Load ACS housing data.  
        housing = pd.read_csv(ACS_HOUSING_PATH % year, skiprows=[1], na_values=["(X)", "-", "**", "2,000+", "3,500+", "1,000,000+", "10,000-", "2,000,000+"])
        # housing = pd.read_csv(ACS_HOUSING_PATH % year, skiprows=[1], na_values=["(X)", "-", "**"])
        # Create ITZ_GEOID column and sort it so it aligns with tract_df index. 
        _add_tract_ids(housing)
        housing.set_index("ITZ_GEOID", inplace=True)
        housing.sort_index(inplace=True)
        # Load dictionary which maps ACS codes to columns
        with open("in-the-zone-data/acs/code-to-column-housing-data-"+str(year)+".txt", "r") as f:
            code_to_column = eval(f.read())
        if year == "2011":
            tract_df["resid_unit_density"] = housing[code_to_column['Estimate!!HOUSING OCCUPANCY!!Total housing units']].astype(float)
            tract_df["percent_multi_family_units"] = 100  \
                - housing[code_to_column['Percent!!UNITS IN STRUCTURE!!1-unit, detached']].astype(float) \
                - housing[code_to_column["Percent!!UNITS IN STRUCTURE!!1-unit, attached"]].astype(float)
            tract_df["percent_occupied_housing_units"] = housing[code_to_column['Percent!!HOUSING OCCUPANCY!!Occupied housing units']].astype(float)
            # 86 tracts have gross rent = '2000+', these are filtered out later
            tract_df["median_gross_rent"] = housing[code_to_column['Estimate!!GROSS RENT!!Median (dollars)']].astype(float)
            # 110 tracts have median house value = '1,000,000+', these are filtered out later
            tract_df["median_home_value"] = housing[code_to_column['Estimate!!VALUE!!Median (dollars)']].astype(float)
        else:
            tract_df["resid_unit_density"] = housing[code_to_column['Estimate!!HOUSING OCCUPANCY!!Total housing units']].astype(float)
            tract_df["percent_multi_family_units"] = 100  \
                - housing[code_to_column['Percent!!UNITS IN STRUCTURE!!Total housing units!!1-unit, detached']].astype(float) \
                - housing[code_to_column['Percent!!UNITS IN STRUCTURE!!Total housing units!!1-unit, attached']].astype(float)
            tract_df["percent_occupied_housing_units"] = housing[code_to_column['Percent!!HOUSING OCCUPANCY!!Total housing units!!Occupied housing units']].astype(float)
            # 86 tracts have gross rent = '2000+', these are filtered out later
            tract_df["median_gross_rent"] = housing[code_to_column['Estimate!!GROSS RENT!!Occupied units paying rent!!Median (dollars)']].astype(float)
            # 110 tracts have median house value = '1,000,000+', these are filtered out later
            tract_df["median_home_value"] = housing[code_to_column['Estimate!!VALUE!!Owner-occupied units!!Median (dollars)']].astype(float)

        del housing

        print("Tract housing data collected")

        # Load ACS social data.  
        social = pd.read_csv(ACS_SOCIAL_PATH % year, skiprows=[1], na_values=["(X)", "-", "**"])
        # Create ITZ_GEOID column and sort it so it aligns with tract_df index. 
        _add_tract_ids(social)
        social.set_index("ITZ_GEOID", inplace=True)
        social.sort_index(inplace=True)
        # Load dictionary which maps ACS codes to columns
        with open("in-the-zone-data/acs/code-to-column-social-data-"+str(year)+".txt", "r") as f:
            code_to_column = eval(f.read())
        if year == "2011":
            tract_df["percent_households_with_people_under_18"] = social[code_to_column['Percent!!HOUSEHOLDS BY TYPE!!Households with one or more people under 18 years']].astype(float)
            tract_df["percent_of_households_in_same_house_year_ago"] = social[code_to_column['Percent!!RESIDENCE 1 YEAR AGO!!Same house']].astype(float)
            tract_df["percent_bachelor_degree_or_higher"] = social[code_to_column["Percent!!EDUCATIONAL ATTAINMENT!!Percent bachelor's degree or higher"]].astype(float)
        elif year == "2019":
            tract_df["percent_households_with_people_under_18"] = social[code_to_column['Percent!!HOUSEHOLDS BY TYPE!!Total households!!Households with one or more people under 18 years']].astype(float)
            tract_df["percent_of_households_in_same_house_year_ago"] = social[code_to_column['Percent!!RESIDENCE 1 YEAR AGO!!Population 1 year and over!!Same house']].astype(float)
            tract_df["percent_bachelor_degree_or_higher"] = social[code_to_column["Percent!!EDUCATIONAL ATTAINMENT!!Population 25 years and over!!Bachelor's degree or higher"]].astype(float)
        else:
            tract_df["percent_households_with_people_under_18"] = social[code_to_column['Percent!!HOUSEHOLDS BY TYPE!!Households with one or more people under 18 years']].astype(float)
            tract_df["percent_of_households_in_same_house_year_ago"] = social[code_to_column['Percent!!RESIDENCE 1 YEAR AGO!!Population 1 year and over!!Same house']].astype(float)
            tract_df["percent_bachelor_degree_or_higher"] = social[code_to_column["Percent!!EDUCATIONAL ATTAINMENT!!Percent bachelor's degree or higher"]].astype(float)

        del social

        print("Tract social data collected")

        # Load ACS transportation data.  
        transportation = pd.read_csv(ACS_TRANSPORTATION_PATH % year, skiprows=[1], na_values=["(X)", "-", "**", "N"])
        # transportation = pd.read_csv(ACS_TRANSPORTATION_PATH % year, skiprows=[1], na_values=["(X)", "-", "**"])
        # Create ITZ_GEOID column and sort it so it aligns with tract_df index. 
        _add_tract_ids(transportation)
        transportation.set_index("ITZ_GEOID", inplace=True)
        transportation.sort_index(inplace=True)
        # Load dictionary which maps ACS codes to columns
        with open("in-the-zone-data/acs/code-to-column-transportation-data-"+str(year)+".txt", "r") as f:
            code_to_column = eval(f.read())
        if year == "2019":
            tract_df["percent_car_commuters"] = 100 * \
                (transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!Workers 16 years and over']].astype(float) + transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!Workers 16 years and over']].astype(float)) \
                / transportation[code_to_column['Estimate!!Total!!Workers 16 years and over']].astype(float)

            tract_df["percent_public_transport_commuters"] = 100 * \
                transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!Workers 16 years and over']] \
                / transportation[code_to_column['Estimate!!Total!!Workers 16 years and over']]

            tract_df["percent_public_transport_trips_under_45_min"] = 100 * \
                (transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!35 to 44 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!30 to 34 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!25 to 29 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!20 to 24 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!15 to 19 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!10 to 14 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!Less than 10 minutes']].astype(float)) \
                / transportation[code_to_column["Estimate!!Public transportation (excluding taxicab)!!Workers 16 years and over"]].astype(float)
            
            tract_df["percent_car_trips_under_45_min"] = 100 * \
                (transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!35 to 44 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!30 to 34 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!25 to 29 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!20 to 24 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!15 to 19 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!10 to 14 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!Less than 10 minutes']].astype(float)+ \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!35 to 44 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!30 to 34 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!25 to 29 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!20 to 24 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!15 to 19 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!10 to 14 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!Workers 16 years and over who did not work from home!!TRAVEL TIME TO WORK!!Less than 10 minutes']].astype(float)) \
                / (transportation[code_to_column["Estimate!!Car, truck, or van -- drove alone!!Workers 16 years and over"]].astype(float) + \
                 transportation[code_to_column["Estimate!!Car, truck, or van -- carpooled!!Workers 16 years and over"]].astype(float))
        else:
            tract_df["percent_car_commuters"] = 100 * \
                (transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!Workers 16 years and over']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!Workers 16 years and over']].astype(float)) \
                / transportation[code_to_column['Estimate!!Total!!Workers 16 years and over']].astype(float)

            tract_df["percent_public_transport_commuters"] = 100 * \
                transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!Workers 16 years and over']] \
                / transportation[code_to_column['Estimate!!Total!!Workers 16 years and over']]

            tract_df["percent_public_transport_trips_under_45_min"] = 100 * \
                (transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!TRAVEL TIME TO WORK!!35 to 44 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!TRAVEL TIME TO WORK!!30 to 34 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!TRAVEL TIME TO WORK!!25 to 29 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!TRAVEL TIME TO WORK!!20 to 24 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!TRAVEL TIME TO WORK!!15 to 19 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!TRAVEL TIME TO WORK!!10 to 14 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Public transportation (excluding taxicab)!!TRAVEL TIME TO WORK!!Less than 10 minutes']].astype(float)) \
                / transportation[code_to_column["Estimate!!Public transportation (excluding taxicab)!!Workers 16 years and over"]].astype(float)
            
            tract_df["percent_car_trips_under_45_min"] = 100 * \
                (transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!TRAVEL TIME TO WORK!!35 to 44 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!TRAVEL TIME TO WORK!!30 to 34 minutes']].astype(float) + \
                transportation[code_to_column["Estimate!!Car, truck, or van -- drove alone!!TRAVEL TIME TO WORK!!25 to 29 minutes"]].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!TRAVEL TIME TO WORK!!20 to 24 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!TRAVEL TIME TO WORK!!15 to 19 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!TRAVEL TIME TO WORK!!10 to 14 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- drove alone!!TRAVEL TIME TO WORK!!Less than 10 minutes']].astype(float)+ \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!TRAVEL TIME TO WORK!!35 to 44 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!TRAVEL TIME TO WORK!!30 to 34 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!TRAVEL TIME TO WORK!!25 to 29 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!TRAVEL TIME TO WORK!!20 to 24 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!TRAVEL TIME TO WORK!!15 to 19 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!TRAVEL TIME TO WORK!!10 to 14 minutes']].astype(float) + \
                transportation[code_to_column['Estimate!!Car, truck, or van -- carpooled!!TRAVEL TIME TO WORK!!Less than 10 minutes']].astype(float)) \
                / (transportation[code_to_column["Estimate!!Car, truck, or van -- drove alone!!Workers 16 years and over"]].astype(float) + \
                 transportation[code_to_column["Estimate!!Car, truck, or van -- carpooled!!Workers 16 years and over"]].astype(float))

        del transportation

        # Divide all density columns by tract area. 
        for tract_id in tract_area_df.index:
            tract_area = tract_area_df.at[tract_id, "area"]
            tract_df["pop_density"][tract_id] /= tract_area
            tract_df["pop_density"][tract_id] *= SQFT_TO_SQKM
            tract_df["resid_unit_density"][tract_id] /= tract_area
            tract_df["resid_unit_density"][tract_id] *= SQFT_TO_SQKM

        print("Tract transportation data collected")

        # Append the tract_df for this year to the list of all tract_dfs. 
        tract_dfs.append(tract_df)

    return tract_dfs

def _add_tract_ids(tract_df: pd.DataFrame):
    """Adds an "ITZ_GEOID" column to the data combining the borough and census tract number.
    """
    # Different possibilities for GEOID. 
    try:
        itz_geoids = []
        for _, row in tract_df.iterrows():
            geoid = row["GEOID10"]
            itz_geoids.append(CODE_TO_COUNTY[geoid[11:14]] + row["NAME"][13:row["NAME"].find(",")])
    except:
        itz_geoids = []
        for _, row in tract_df.iterrows():
            geoid = row["GEO_ID"]
            itz_geoids.append(CODE_TO_COUNTY[geoid[11:14]] + row["NAME"][13:row["NAME"].find(",")])

    tract_df["ITZ_GEOID"] = itz_geoids


def _get_lot_data() -> pd.DataFrame:
    """Creates DataFrame with columns being lot-specific data and rows being lots. 
    Index: ITZ_GEOID
    Uses 2011 Lots - all other lots are discarded. 
    """
    # 2011 is a starting year and uses 2000 census tracts- so next_pluto, or 2012 data, has to be loaded in order to create
    # ITZ_GEOIDs correctly. 
    try:
        starting_pluto = pd.read_csv(PLUTO_PATH % LOT_DATA_YEARS[0], header=0, dtype=str)
    except:
        starting_pluto = pd.read_table(PLUTO_TEXT_PATH % LOT_DATA_YEARS[0], header=0, sep=",", dtype=str, usecols=["BBL", "Version", "Borough", "LotArea"])
        next_pluto = pd.read_table(PLUTO_TEXT_PATH % 2012, header=0, sep=",", dtype=str, usecols=["BBL", "CT2010"])
        print("next pluto created")
        next_pluto.set_index("BBL", inplace=True)
    print("starting pluto created")

    starting_pluto.set_index("BBL", inplace=True)
    starting_pluto.sort_index()
    print(starting_pluto)

    # Create ITZ_GEOID column in lot_df
    if starting_pluto["Version"].iloc[0] == "11v1  ":
        print(next_pluto)
        starting_pluto = starting_pluto.join(next_pluto, on="BBL")
        del next_pluto
        print(starting_pluto)
        print(starting_pluto.columns)
        starting_pluto["ITZ_GEOID"] = starting_pluto["Borough"] + starting_pluto["CT2010"].str.strip()
    else: 
        starting_pluto["ITZ_GEOID"] = starting_pluto["Borough"] + starting_pluto["CT2010"]

    # Filter starting_pluto for valid ITZ_GEOIDs
    starting_pluto = starting_pluto[starting_pluto['ITZ_GEOID'].map(lambda x: len(str(x)) != 2)]
    starting_pluto = starting_pluto[starting_pluto["ITZ_GEOID"].notnull()]
    # Copy the index of BBLs in starting_pluto
    lot_bbl = starting_pluto.index

    # Create the list of columns in lot_df. 
    columns = []
    for year in LOT_DATA_YEARS:
        columns.append("land_use"+year)
        columns.append("zoning"+year)
        columns.append("max_resid_far"+year)
        columns.append("mixed_development"+year)
        columns.append("limited_height"+year)
        columns.append("resid_units"+year)
    # Create lot_df. 
    lot_df = pd.DataFrame(index=lot_bbl, columns=columns)
    # Copy the ITZ_GEOIDs from starting_pluto. 
    lot_df["ITZ_GEOID"] = starting_pluto["ITZ_GEOID"]
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
        # Set the index and sort by BBL in order to maintain consistency
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
        # print(pluto_df)
        try:
            lot_df["land_use" + year] = pluto_df["LandUse"]
            print(100*len(pluto_df[(pluto_df["LandUse"] == "01") | (pluto_df["LandUse"] == "02") | (pluto_df["LandUse"] == "03") | (pluto_df["LandUse"] == "04") | (pluto_df["LandUse"] == "1") | (pluto_df["LandUse"] == "2") | (pluto_df["LandUse"] == "3") | (pluto_df["LandUse"] == "4")])/len(pluto_df), "percentage!")
        except:
            lot_df["land_use" + year] = pluto_df["landuse"]
            print(100*len(pluto_df[(pluto_df["landuse"] == "01") | (pluto_df["landuse"] == "02") | (pluto_df["landuse"] == "03") | (pluto_df["landuse"] == "04") | (pluto_df["landuse"] == "1") | (pluto_df["landuse"] == "2") | (pluto_df["landuse"] == "3") | (pluto_df["landuse"] == "4")])/len(pluto_df), "percentage!")
        lot_df["land_use"+year].to_csv("land_use"+year+".csv")
            # print(len(pluto_df[pluto_df["landuse"] < 5])/len(pluto_df), "percentage!")
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
        try:
            lot_df["mixed_development" + year] = (pluto_df["landuse"] == "4") | (pluto_df["landuse"] == "04")
            print(pluto_df.loc[(pluto_df["landuse"] == "4") | (pluto_df["landuse"] == "04")]["landuse"], " is there mixed development")
        except KeyError:
            lot_df["mixed_development" + year] = (pluto_df["LandUse"] == "4") | (pluto_df["LandUse"] == "04")
            print(pluto_df.loc[(pluto_df["LandUse"] == "4") | (pluto_df["LandUse"] == "04")]["LandUse"], " is there mixed development")
        lot_df["mixed_development"+year].to_csv("mixed_development"+year+".csv")
        print(lot_df.loc[lot_df["mixed_development2011"] == True])

        try:
            lot_df["limited_height" + year] = pluto_df["ltdheight"].str.strip() != ''
        except KeyError:
            lot_df["limited_height" + year] = pluto_df["LtdHeight"].str.strip() != ''
        try:
            lot_df["resid_units" + year] = pluto_df["unitsres"]
        except KeyError:
            lot_df["resid_units" + year] = pluto_df["UnitsRes"]
        lot_df["resid_units"+year].to_csv("resid_units"+year+".csv")
        lot_df["mixed_development"+year].to_csv("mixed_development"+year+".csv")

        del pluto_df
    print("Lot Data year data collected!")
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
        "orig_percent_subsidized_properties",
        "orig_percent_multi_family_units",
    ])
    for delta in DELTAS:
        print("Now working on: ", delta)
        start = delta[0]
        end = delta[1]
        for tract, lot_list in tracts_to_lots.items():
            # print(tract, lot_list)
            upzoned = 0
            years_since_upzoned = []
            lots_with_res_unit_data = 0
            residential_units_start = 0
            residential_units_end = 0
            for lot in lot_list:
                prev = float(lot_df.at[lot, "max_resid_far"+str(start)])*float(lot_df.at[lot, "lot_area"])
                for year in range(int(start)+1, int(end)+1):
                # for year in [2011, 2016, 2019]:
                    curr = float(lot_df.at[lot, "max_resid_far"+str(year)])*float(lot_df.at[lot, "lot_area"])
                    if prev != 0 and curr/prev > 1.1:
                        upzoned += 1
                        years_since_upzoned.append(int(end)-int(year))
                        break
                    prev = curr
                try:
                    _ = int(lot_df.at[lot, "resid_units"+str(start)])
                    _ = int(lot_df.at[lot, "resid_units"+str(end)])
                except:
                    continue
                else:
                    residential_units_start += int(lot_df.at[lot, "resid_units"+str(start)])
                    residential_units_end += int(lot_df.at[lot, "resid_units"+str(end)])
                    lots_with_res_unit_data += 1
                    # print(residential_units_start, residential_units_end)
                try:
                    _ = float(residential_units_end)
                except:
                    raise Exception()
            tract_lot_data.at[tract, start + "_" + end + "_percent_upzoned"] = 100 * upzoned/len(lot_list)
            if len(years_since_upzoned) != 0:
                tract_lot_data.at[tract, start + "_" + end + "_average_years_since_upzoning"] = sum(years_since_upzoned)/len(years_since_upzoned)
            tract_lot_data.at[tract, "d_" + start + "_" + end + "_resid_units"] = residential_units_end-residential_units_start


    for tract, lot_list in tracts_to_lots.items():
        land_use_lots = 0
        residential = 0
        limited_height = 0
        mixed_development = 0
        for lot in lot_list:    
            # 1 corresponds to one/two family, 2 and 3 correspond to multi-family, 4 corresponds to mixed_resid/comm
            # try:
            if lot_df["land_use2011"][lot].strip() != '':
                if int(lot_df["land_use2011"][lot]) < 5:
                    residential += 1
                land_use_lots += 1
                # print(lot_df["mixed_development2011"][lot])
                # print(type(lot_df["mixed_development2011"][lot]))
                if lot_df["mixed_development2011"][lot].astype(bool) == True:
                    mixed_development += 1
            # except ValueError:
                # pass
            if lot_df["limited_height2011"][lot] == True:
                limited_height += 1
        try:
            tract_lot_data.at[tract, "orig_percent_residential"] = 100 * residential/land_use_lots
            tract_lot_data.at[tract, "orig_percent_mixed_development"] = 100 * mixed_development/land_use_lots
        except:
            print("TRACT WITH NO LAND USE LOTS??", tract)
            tract_lot_data.at[tract, "orig_percent_residential"] = 0
            tract_lot_data.at[tract, "orig_percent_mixed_development"] = 0
        tract_lot_data.at[tract, "orig_percent_limited_height"] = 100 * limited_height/len(lot_list)

    subsidized_property_by_tract = pd.read_csv(SUBSIDIZED_PROPERTIES_PATH)["tract_10"].value_counts(sort=False)
    subsidized_property_by_geoid = {}
    for tract, count in subsidized_property_by_tract.items():
        str_tract = str(tract)
        if str_tract[-1] != "0":
            itz_geoid = CODE_TO_COUNTY[str_tract[2:5]]+str(int(str_tract[5:9]))+"."+str(int(str_tract[9:]))
        else:
            itz_geoid = CODE_TO_COUNTY[str_tract[2:5]]+str(int(str_tract[5:9]))
        subsidized_property_by_geoid[itz_geoid] = count
    for itz_geoid, lot_list in tracts_to_lots.items():
        try:
            tract_lot_data.at[itz_geoid, "orig_percent_subsidized_properties"] = 100 * subsidized_property_by_geoid[itz_geoid]/len(lot_list)
            del subsidized_property_by_geoid[itz_geoid]
        except:
            tract_lot_data.at[itz_geoid, "orig_percent_subsidized_properties"] = 0
    print(subsidized_property_by_geoid)
    return tract_lot_data


def _get_delta_data(tract_dfs, index) -> pd.DataFrame:
    """Calculates the changes for tract-specific data between starting and ending points. 
    """
    tract_delta = pd.DataFrame(index=index)
    date_to_index = {'2011':0, '2016':1, '2019':2}

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
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_percent_public_transport_trips_under_45_min"] = tract_dfs[end_index]["percent_public_transport_trips_under_45_min"] - tract_dfs[start_index]["percent_public_transport_trips_under_45_min"]
        tract_delta["d_"+delta[0]+"_"+delta[1]+"_percent_car_trips_under_45_min"] = tract_dfs[end_index]["percent_car_trips_under_45_min"] - tract_dfs[start_index]["percent_car_trips_under_45_min"]
        
    return tract_delta