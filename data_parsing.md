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

## Data columns (before delta conversions)

### Columns to add to data

#### Density columns

Divide these by tract area (data source TBD)

- pop_density: DP05_0001E
- resid_unit_density: DP04_0001E

#### Others

- multi_family_units: 100 - DP04_0007PE
- percent_car_commuters: (S0802_C02_001E + S0802_C03_001E) / S0802_C01_001E
- percent_public_transport_commuters: S0802_C04_001E / S0802_C01_001E
- percent_resid: % of lots in tract for which ResUnits > 0
- percent_ltd_height: % of lots in tract for which LtdHeight is not just spaces

### Name changes

- DP04_0088PE: median_housing_price
    - Not sure exactly what this is measuring. Description: "Percent!!VALUE!!Median (dollars)" - from ACS housing data
- S0802_C04_090E: mean_public_transport_travel_time
- S0802_C03_090E: mean_car_travel_time
- DP05_0072PE: percent_non_hispanic_or_latino_white_alone
- DP03_0063M: mean_income

## Delta conversions

Create a new dataframe (assuming previous work was done with each year having a separate dataframe)
Convert each of these data columns into "d_"-prefixed columns as change measures.

## 2011 value controls

Add a column for the 2011 values for these (prefix with orig_).

mean_income, pop_density, percent_non_hispanic_or_latino_white_alone

## More metrics

### % Upzoned

For each tract:
    calculate the percentage of lots in the tract for which MaxAllwFAR * LotArea has increased by more than 10% between the start and end years.

Add as a column called percent_upzoned