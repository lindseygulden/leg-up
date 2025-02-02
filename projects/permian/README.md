## Oil and gas production data
We queried publicly available oil and gas production data from the states of New Mexico and Texas to estimate ExxonMobil's Permian basin petroleum production. 

### Texas
From the Texas Railroad Commission's public database, we queried operator-total monhtly production for each month from January 2017 through November 2024 for both XTO ENERGY INC. and PIONEER NATURAL RES. USA, INC. We queried production for the four oil and gas districts in Texas that contain 'Permian' acreage (8, 8A, 7B, and 7C). For districts 8, 8A, 7B, and 7C, ExxonMobil was not listed under any registration as an operator/producer. A map of the districts can be found at 

ExxonMobil does business in the Texas Permian as the Exxon Mobil Corp subsidiary XTO ENERGY INC. Pioneer, which does business as 'PIONEER NATURAL RES. USA, INC' in Texas, only had production reported in districts 8, 8A, and 7C (none in 7B during the time period queried).
Data are publicly accessible at http://webapps.rrc.texas.gov/PDQ/generalReportAction.do. For each operator, and each district, we queried the district-total production for each month since January 2017 and assembled the data into a single dataframe.

In 'monthly' results for the General Production Query, the Texas Railroad Commission reports the following volumes:
1. Oil produced, in units of barrels (bbl)
2. Casinghead (CH) gas (i.e., gas disolved in crude oil or condensate), in units of thousands of cubic feed (Mcf)
3. Gas well (GW) gas, which is gas produced in wells for which the dominant petroleum product is natural gas rather than oil, in units of Mcf.
4. Condensate, which is effectively 'light' oil, in units of bbls

### New Mexico
From the New Mexico Oil Conservation Division (OCD), we obtained the production data for the two Exxon Mobil Corporation subsidiaries operating in the state of New Mexico, XTO ENERGY INC and XTO PERMIAN OPERATING LLC. Pioneer produced nothing in the state of NM between the years 2019 and 2024. Monthly oil production data (in units of bbls) and gas production data (in units of MCF) data are reported in the C-115 Summary Balancing Reports, which are publicly accessible at https://wwwapps.emnrd.nm.gov/OCD/OCDPermitting/Reporting/Production/C115BalancingSummary.aspx. The reports provide statewide total production for a given operator. As of Feburary 2, 2025, data were available for all months in years 2019-2023 and for the first 11 months of 2024.

The New Mexico OCD reports the following volumes for a given operator for each month listed in the C-115 Summary Balancing Reports:
1. Oil produced, in units of barrels (bbl)
2. Gas produced, in units of thousands of cubic feet (Mcf)
3. Additional information about the volume of oil and gas transported, as well as the 'variance' (?)

## How did you define the 'Permian'?
We used ExxonMobil's [maps of its Permian acreage](https://corporate.exxonmobil.com/who-we-are/our-global-organization/business-divisions/upstream/unconventional), combined with the [Texas Railroad Commission's maps of oil and gas districts](https://rrc.texas.gov/media/3bkhbut0/districts_color_8x11.pdf) to help identify which RRC districts span the Permian.

The most accessible New Mexico data--the C-115 summary reports--provide production data that are aggregated across the entire state. In the state of New Mexico, it appears that 'XTO Permian Operating LLC' (not 'XTO Energy Inc.') is the operator that handles Permian assets. XTO Permian Operating LLC began reporting production in mid 2019, which is when ExxonMobil/XTO began producing in Eddy and Lea counties in earnest. We assume here that data reported for 'XTO Permian Operating LLC' coincide with Exxon Mobil Corporation's Permian production in New Mexico. Pioneer does not operate in New Mexico.

## Data processing
To enable comparison to the October 11, 2023 ExxonMobil press release, we converted all gas production volumes for both Texas and New Mexico to 'barrels of oil equivalent' (boe) by dividing the number of thousand cubic feed of gas by 6, as is customary.

## Observations
1. In an October 11, 2023, press release announcing its acquisition of Pioneer Natural Resources, ExxonMobil stated, "At close, ExxonMobil’s Permian production volume would more than double to 1.3 million barrels of oil equivalent per day (MOEBD), based on 2023 volumes, and is expected to increase to approximately 2 MOEBD in 2027." We used this assertion to confirm that the data we collected are consistent with ExxonMobil's view of its own production.


