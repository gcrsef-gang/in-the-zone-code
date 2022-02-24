# Data Parsing

Note: income data is inflation-adjusted

## ACS Data Prefixes

- S0802 - transportation
- DP05 - demographic
- DP03 - economic
- DP04 - housing
- DP02 - social

## Matching lots to tracts

Create a new column in ACS and PLUTO data called ITZ_GEOID, combining the borough and the tract
number. (ex. "BX1" for Bronx census tract 1.)

If a PLUTO year has 2000 tracts instead of 2010, map to 2010 using
`2000-to-2010-census-blocks-tracts.txt`

Create a dictionary of census tracts to the lots they contain.

## Lot data processing

Filter by whether the lot is a residential lot using LandUse. 
Create lot_df which tells us whether it's been upzoned, and the number of years since. 

### Columns

- MaxAllwFAR/ResidFAR
- LotArea
- ResUnits
- LtdHeight
- LandUse

## Tract data processing (before delta conversions)

Create one dataframe for each of the start/end years.

### Columns not requiring lot data

#### Density columns

Find tract area with GEOID10 in ny_2010_census_tracts.json.
Divide these by tract area (ALAND10). 
- pop_density: DP05_0001E
- resid_unit_density: DP04_0001E

#### Others

- % multi_family_units: 100 - (DP04_0007PE+DP04_0008PE) (subtracting one unit attached/detached houses)
- percent_car_commuters: (S0802_C02_001E + S0802_C03_001E) / S0802_C01_001E
- percent_public_transport_commuters: S0802_C04_001E / S0802_C01_001E

### Name changes

- DP04_0088E: median_home_value
    - Not sure exactly what this is measuring. Description: 'Estimate!!VALUE!!Median (dollars)' - from ACS housing data
    - This actually measures the median home value in the census tract. - Jam
- S0802_C04_090E: mean_public_transport_travel_time
- S0802_C03_090E: mean_car_travel_time
- DP05_0072PE: percent_non_hispanic_or_latino_white_alone
- DP05_0073PE: percent_non_hispanic_black_alone
- DP05_0066PE: percent_hispanic_any_race
- DP05_0075PE: percent_non_hispanic_asian_alone
- DP03_0088E: per_capita_income
- DP05_0017E: median_age
- DP02_0013PE: percent_households_with_people_under_18
- DP04_0002PE: percent_occupied_housing_units
- DP04_0132E: median_gross_rent
- DP02_0079PE: percent_of_households_in_same_house_year_ago
- DP02_0067PE: percent_bachelor_degree_or_higher

### Columns requiring lot data

- percent_resid: % of lots in tract for which ResUnits > 0
- percent_ltd_height: % of lots in tract for which LtdHeight is not just spaces

## Delta conversions

Create a new dataframe (assuming previous work was done with each year having a separate dataframe)
Convert each of these data columns into "d_"-prefixed columns as change measures.

## 2011 value controls

Add a column for the 2011 values for these (prefix with orig_).

per_capita_income, pop_density, percent_non_hispanic_or_latino_white_alone, percent_non_Hispanic_black_alone, percent_hispanic_any_race, percent_non_hispanic_asian_alone, percent_occupied_housing_units, median_age, 
percent_households_with_people_undr_18, percent_of_households_in_same_house_year_ago, percent_bachelor_degree_or_higher

## More metrics

### % Upzoned

For each tract:
    calculate the percentage of lots in the tract for which MaxAllwFAR/ResidFAR * LotArea has increased by more than 10% between the start and end years.

Add as a column called percent_upzoned

### Land Use Entropy 

Captures the homogeneity/heterogeneity of the census tract - 0 represents a tract where every lot has the same land use,
while 1 represents a tract where land uses are evenly split between lots. 

Calculated as -1 * sum of p_j * ln(p_j) over all j, where j is the number of land uses and p_j is the proportion of lots
with land use j. 

### Percent of properties which are subsidized 

For each lot in tract: 
Check if in subsidized housing records in subsidized_properties.csv using BBL number. (i.e. if lot is subsidized and before 3/1/2011)
Create variable orig_percent_subsidized_properties. 

### Average # of years since upzoned

Average for each of the lots in the tract, set to 0 if no upzoning occurred.