"""Data parsing for the American Consumer Survey and NYC PLUTO databases.
"""

# Whether ACS data uses 2000 census geo units instead of 2010
OLD_BLOCKS = ()  # Tuple of bools
ACS_PATH = ""
PLUTO_PATH = ""

VAR_NAMES = ()

def get_data():
    """Creates DataFrame with columns corresponding to variables used in SEMs.
    """
    # Read CSVs
    # Change ACS data to change-based columns
    # Assign lots to tracts
    # Get upzoning data for each tract


def _get_tract_dict(lot_df, old_blocks=False, block_dict=None, tract_dict=None):
    """Creates a dictionary mapping 2010 census tracts to the set of lots they hold.

    Parameters
    ----------
    lot_df: pd.DataFrame
        Lot-based zoning data from PLUTO
    old_blocks: bool (optional)
        Whether `lot_df` uses 2000 census geo units instead of 2010
    block_dict: dict (optional)
        Maps 2000 census block IDs to 2010 block IDs
    tract_dict: dict (optional)
        Maps 2010 census blocks to census tracts
    """


def _get_upzoning(lot_df, tracts):
    """Calculates the percent of each tract that was upzoned using the (using a 10% threshold in
    maximum residential capacity)
    """