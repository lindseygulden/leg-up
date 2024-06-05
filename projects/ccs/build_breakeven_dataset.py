""" Script for constructing a dataframe of breakeven price distribuition parameters
to be used as part of probabilistic unit-economics simulations"""

# ---- METHODS ----
# Data are based on figures presented as part of the Dallas Fed Energy Survey dated
# March 2024. The Dallas Fed asked energy excutives what their companies' oil-breakeven
# prices were for new wells and existing wells across the US. The Dallas Fed presented
# results in two figures, found here: https://www.dallasfed.org/research/surveys/des/2024/2401#tab-questions
# We used the two figures to construct triangular distributions to represent likely breakeven prices
# for new and existing wells in the US.
# To compute the 'mode' parameter for numpy's random.triangular distribution, we found the US-mean
# breakeven price (for both new and existing wells) by finding the weighted mean value for each of the
# two figures. Specifically
# for existing wells: (31*22+18*34+5*35+38*24+18*43+45*50)/(22+18+5+24+18+50) = 39.5
# for new wells: (59*17+23*62+64*21+5*65+66*45+18*70)/(17+23+21+5+45+18) = 64.6
# For the 'left' and 'right' parameters (representing the lowest and highest values of the triangular
# distribution, respectively), we used the lowest and highest values from digitized versions of the figures
# linked above.
#


import pandas as pd


def breakeven(output_file=None, conversion_factor=0.97):
    """Generates, writes out, and returns a dataframe with parameters for triangular distributions for
    oil breakeven prices"""
    breakeven_prices_df = (
        pd.DataFrame(
            {"low": [5, 30], "mid": [39.5, 64.6], "high": [90, 95]},
            index=["existing", "new"],
        )
        * conversion_factor  # convert to a given year's dollars
        # default value of 0.97 converts 2024 dollars to 2023 dollars
    )
    if output_file is not None:
        breakeven_prices_df.to_csv(output_file)
    return breakeven_prices_df


if __name__ == "__main__":
    breakeven()
