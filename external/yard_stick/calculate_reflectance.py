"""
Dear Hypo,
Thanks for the opportunity to review this script. Following are a few comments/questions/suggestions.
I've also added a few minor suggestions/questions throughout the code itself with the tag '#LEG'.
I'm happy to hop on a call to discuss any/all if that would be useful to you.
Lindsey

1.  Comments regarding calibration data:
    a.  Requirements say "most recent calibration data for the device at the time of the scan", which I interpret
        to mean that the calibration date must be the latest calibration that occurred BEFORE the scan date. 'Suggest adding
        an additional 'WHERE' clause in the calibration SQL query to find calibrations whose dates are <= the scan's date.
        See https://www.sqlite.org/lang_datefunc.html for more info on representing dates.
    b.  To increase re-usability of the script, consider changing the version to a string variable read in from the 'device' column
        of the dataframe read in from the scans.json data
2.  Comments regarding data and wavelength:
    a.  The length of each DN array in scans.json is 1920; the length of the latest 4v9 calibration arrays are 1845. Given
        these array-length mismatches, the script doesn't run as written. To fix: I presume we need to know
        which observations in the calibration data correspond to which wavelengths, which is a nice segue to the next comment...
    b.  Do we have access to additional metadata from the calibrations that will allow us to assign the calibrated shutter_reflectance
        and stray_reflectance values to specific wavelengths? (So far as I can tell, the calibration data are
        not labeled by wavelength -- 'SELECT * FROM calibrations' yields no additional info.) I'm assuming we can treat the order
        of the 'wavelength' array in scans.json as labels for the corresponding 'DN' array in a given row, but it is not clear
        to me how to align the scan data and the calibration data when the lengths of the arrays differ (as is the case here).
        Assuming these wavelength data are available in a different table, to guard against error, consider adjusting the script
        to ensure pairing of same-wavelength data.
3.  Comments regarding computation of reflectance:
    a.  As written, the formula for reflectance of a shutter-open scan does not seem to match what is listed in the requirements.
        I've provided my interpretation of the requirements inline in the code below.
    b.  Is there ever a situation in which dn_shutter_closed is 0? If so, the equation provided in the requirements will give a 'divide by zero' error
4.  General structure/style/implementation comments:
    a.  Consider turning this script into a module/function(s) such that it can either be called via the command line or programmatically;
        functionalizing the capability will also make it easier to implement error checking (e.g., poorly formatted data, missing data, etc.)
    b.  Do we have any already-computed results that we could use to build a unit test for this script?
    c.  Depending on how frequently we'd like to use the script: consider turning this into a command-line script with
        the click library (https://click.palletsprojects.com/en/8.1.x/)
    d.  The structure below works if a given scan.json dataset does not contain scans from different devices and/or different dates;
        if that may change in the future, consider restructuring the code to identify the proper calibration data for each row of the
        scans dataframe
    e.  Ideally include an explanation of the code, module docstring, etc. Perhaps include the details presented in the requirements
    f.  Nit: depending on (e) above, consider renaming variables (in code and/or documentation) to match each other
"""

import json
import sqlite3

import numpy as np
import pandas as pd

scans = pd.read_json("scans.json", orient="records")

# Average out closed scans
shutter_closed_scans = scans[scans["shutter"] == "CLOSED"]
averaged_shutter_closed_scan = []
for wavelength in zip(*shutter_closed_scans["DN"]):
    # LEG For next line, is there a reason you're not using np.nanmean or np.mean?
    average_for_wavelength = sum(wavelength) / len(wavelength)
    averaged_shutter_closed_scan.append(average_for_wavelength)

# Fetch most recent calibration
# LEG See comments 1a and 1b, above
# LEG suggest making query an f string, introducing two string variables, device_id and scan_date,
# LEG and changing WHERE clause to WHERE device_id = '{device_id}' AND calibrated_at <= date('{scan_date}')
calibration = (
    pd.read_sql(
        f"""
        SELECT shutter_reflectance, stray_reflectance
        FROM calibrations
        WHERE device_id = '4v9'
        ORDER BY calibrated_at DESC
        LIMIT 1
        """,
        sqlite3.connect("calibrations.sqlite"),
    )
    .applymap(json.loads)
    .iloc[0]
)

# Calculate reflectance for each scan
scans["reflectance"] = None

# LEG Especially if scanning datasets will become lengthy/numerous,
# LEG consider a more efficient implementation of this loop (e.g. apply. see https://www.learndatasci.com/solutions/how-iterate-over-rows-pandas/)
for i, scan in scans.iterrows():
    # LEG Replace this with something similar to what you did above? (e.g., open_scans=scans[scans['shutter']=='OPEN'])
    if scan["shutter"] != "OPEN":
        continue

    dn_shutter_open = np.array(scan["DN"])
    # LEG for efficiency, consider moving the next three lines, which don't change, outside/ahead of this for loop
    dn_shutter_closed = np.array(averaged_shutter_closed_scan)
    shutter_reflectance = np.array(calibration["shutter_reflectance"])
    stray_reflectance = np.array(calibration["stray_reflectance"])
    # LEG The next line appears to not match the requirements. A proposed adjustment follows.
    scans.at[i, "reflectance"] = (dn_shutter_open / dn_shutter_closed) * (
        shutter_reflectance - stray_reflectance
    )
# LEG proposed adjustment to previous line:
#    scans.at[i, "reflectance"] = (dn_shutter_open / dn_shutter_closed) * (
#        shutter_reflectance + stray_reflectance
#    ) - stray_reflectance


scans.to_json("scans_with_reflectance.json", orient="records")
