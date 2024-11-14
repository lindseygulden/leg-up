| column name | explanation | Additional notes on source data, data processing |
| ----------- | ----------- | ----------- |
| filing_uuid | unique identifier for the filing document parsed | Note that, to find the filing document refereneced, this filing_uuid can be entered in the web query interface at lda.senate.gov. |
| total_number_lobbying_activities | total number of individual lobbying activies reported on the filing document | |
| total_number_of_lobbyists_on_filing | total number of lobbyists on the filing document. | Note this is **not** total number of UNIQUE lobbyists b/c this is used to help allocate dollars spent to individual lobbying activities: consider it more of a measure of total lobbying work effort reported on this filing document |
| n_lobbyists_for_activity | number of lobbyists assigned to this particular lobbying activity on the filing document (used to allocate dollars to this activity) | |
| filing_year | year in which the lobbying disclosure document was filed | |
| filing_period | period covered by this document (half/semiannual before 2008; quarterly thereafter) | |
| filing_dt_posted | Date on which the associated disclosure document was filed. Note that, for every unique client_id/registrant_id/filing_period combination, I used the most-recently-filed disclosure | |
| which_congress | Which congress (e.g., 117th) was in session during this period | |
| party_controlling_house | Which party controlled the House for the majority of this Congress | |
| party_controlling_senate | Which party controlled the Senate for the majority of this Congress | |
| party_controlling_white_house | Which party controlled the White House for the majority of this Congress | |
| filing_type | Classification of type of filing. | |
| client_id | 'Unique' identifier for clients--i.e., the organization doing the lobbying--in the LDA database. Note that different subsidiaries of an individual corporation can have different client ids, and it is possible that the client ID can change (?) | |
| organization | 'Cleaned' name of organization doing lobbying ('client'). For example, client_name_on_filing_document EXXON, EXXON MOBIL, and EXXONMOBIL were all changed to be EXXONMOBIL for this field  | client_name_on_filing_document, 'client_name' in LDA API |
| client_name_on_filing_document | Name of the organization funding the lobbying present on the disclosure form itself | |
| client_general_description_on_filing | Description of the client (i.e., the organization doing the lobbying), if any, on the disclosure form | |
| sector | Most granular sector assignment of the organization. Note that some sector assignmnts are obvious; others, less so. For example, NextEra gets ~40% of its profit from fossil fuels...is it in the 'renewable energy' sector? For the gray areas, I made a judgement call. | |
| lumped_sector | a less-fine grained sector assignment | |
| very_lumped_sector | a still-less-fine-grained sector assignment | |
| definitely_ccs | 1 if this lobbying activity is almost surely CCS; 0 otherwise. Lobbying activites were identified as definitely CCS if (1) The activity description mentioned CCS or its many euphemisms (e.g, 'capture and sequestration of carbon dioxide','clean hydrogen','45Q', 'EOR', etc) by name, (2) The activity description mentioned a bill or law that was mostly CCS focused (e.g., the 'USE IT Act' or the 'SCALE Act'), or (3) The disclosure was filed by a company whose entire business is CCS. Note that all of the analysis presented in summaries and figures uses ONLY the subset of activities for which definitely_ccs is True (1) | |
| very_likely_ccs | 1 = True; 0 = False.  | |
| likely_ccs | 1 = True; 0 = False.  | |
| potentially_ccs | 1 = True; 0 = False.  | |
| not_ccs | 1 = True; 0 = False. 1 means this is definitely NOT CCS | |
| registrant_id | Unique identifier for registrants (i.e., the lobbying organization). I am not sure if a single lobbying organization can have more than one registrant_id. Note that, as of Oct. 2024, I also reported details for all these lobbyists, but I have not parsed them yet | |
| who_is_lobbying | Field details whether the organization who is filing the disclosure is the organization lobbying on its own behalf, a lobbying firm, or there is general confusion b/c no dollars are reported | |
| usd_for_all_activities_on_filing_document | **DO NOT** USE TO SUM MONEY (will result in double counting): This value is the total of dollars FOR ALL LOBBYING ACTIVITIES ON THIS FILING DOCUMENT. For summations, use lobbying_activity_usd instead | |
| lobbying_activity_usd | **USE THIS TO SUM MONEY**: This value is the portion of the total dollars spent lobbying reported on the associated filing document that are apportioned to this specific lobbying activity | usd_for_all_activities_on_filing_document, apportioned by fraction of total number of lobbyists on this LDA filing assigned to this specific activity |
| n_entities_lobbied | For this lobbying activity, the number of federal agencies listed as lobbying targets (translates into what I call 'lobbying contacts' in reports) | This is the count of distinct government agencies identified as targets for this lobbying activity |
| legistlative_entities_lobbied | For this lobbying activity, the count/number of legistative-branch entities contacted | Note that, even if every single member of the US House of Representatives is contacted as part of this lobbying activity, this value would still be 1. The value should be interpreted as 'at least one legislative entity lobbying contact was made during this quarter' |
| executive_entities_lobbied | For this lobbying activity, the count/number of executive-branch entities contacted | For example, if both the DOE and EPA were contacted, this field would have a value of 2 |
| affiliated_organizations_present | True if the lobbying disclosure form said 'true' for the question 'were affiliated organizations preesent?' | |
| general_issue_code | The lobbying disclosure form's reporting of the general issue covered by the lobbying activities | |
| activity_description_on_filing | The raw text describing the lobbying activity, as recorded on the filing document. | Note that, for machine/computer searching, this was processed and recorded in cleaned_activity_description column |
| posted_by_name | I assume this is the name of the employee who posted the disclosure form to the goverment | |
| registrant_name | name of lobbying firm (if company is lobbying for itself, not via an external lobbying firm, this is usually the same as the 'client name') | |
| registrant_contact_name | name of lobbying firm's point of contact | |
| client_state | Presumably the state in which the client is located. I can't figure out what the difference between this column and client_ppb_state is | |
| client_ppb_state | Presumably the state in which the client is located. | I can't figure out what the difference between this column and client_state is |
| client_country | Presumably the state in which the client is located. | I can't figure out what the difference between this column and client_ppb_country is. |
| client_ppb_country | reported on LDA form. presumably the state in which the client is located. | I can't figure out what the difference between this column and client_country is. |
| url | url for the filing instance on the lda.senate.gov website | |
| entities_lobbied | A list of the government entities contacted as part of this lobbying activity | In the postprocessed_filings.csv file, this list has been expanded into additional biniary columns, each of which are named with a single entity, for which I don't provide 'decoder' information |
| cleaned_activity_description | Cleaned version (for machine searching) of description of this lobbying activity provided in the filing document | cleaned version (made all lowercase, newlines removed, etc.) of activity_description_on_filing |
| cleaned_client_general_description | Cleaned version (for machine searching) of description of client. Note that many are left blank | |
| contains_description | 1 if the activity contains clear description of CCS (or one of its many related topics/euphemisms: e.g., 'CO2 pipelines', '45Q', 'CCUS', carbon dioxide sequestration, etc.) | |
| count_contains_description | Count of the number of CCS descriptions in the lobbying activity | |
| clean_h2_description | 1 if contains words that typically are associated with clean hydrogen (e.g., the activity description contains both 'hydrogen' and 'methane'); 0 otherwise | |
| count_clean_h2_description | Count of the number of clean hydrogen descriptions in the lobbying activity | |
| ccs_company |  | Intermediate data processing column |
| clean_hydrogen_company | 1 if organization's primary business/focus is 'clean' hydrogen (aka, hydrogen generated using methane/fossil fuels, either as feedstock or as fuel); 0 otherwise | Intermediate data processing column |
| green_hydrogen_company | 1 if organization's primary business/focus is green hydrogen (aka, hydrogen generated NOT with fossil energy, either as fuel or feedstock), 0 otherwise | Intermediate data processing column |
| h2_mention_core_ff | 1 if lobbying activity mentions hydrogen ('hydrogen' or 'h2'), and the lobbying organization is one that directly profits from selling or burning fossil fuels; 0 otherwise | 1 if activity contains terms_consistent_with_ccs and the lobbying organization is in the core FF sector | For example, If Chevron talks about Hydrogen, we assume they're talking about 'clean' hydrogen (not green hydrogen), and we define the activity as CCS. |
| h2_mention_ff_adjacent | 1 if lobbying activity mentions hydrogen ('hydrogen' or 'h2'), and the lobbying organization is one that indirectly profits from selling or burning fossil fuels; 0 otherwise | |
| ccs_bills | 1 if activity explictly mentions bills/laws that were mostly or 100% CCS focused (e.g., USE IT Act); 0 otherwise | |
| bills_with_some_ccs | 1 if activity explictly mentions bills/laws that contained some CCS provisions but were not mostly focused on CCS; 0 otherwise | |
| count_bills_with_some_ccs | Count of the number of bills mentioned in the lobbying activity that have at least some CCS provisions | |
| bills_with_ccs_terms | 1 if a (major) bill with CCS provisions is mentioned along with additional words that indicate the interest is in CCS (e.g., 'Inflation Reduction Act' and 'section 45'); 0 otherwise | |
| ccs_bills_number_only | 1 if activity description contains a bill reference (e.g., 'HR 1231') that was a 'mostly CCS' bill for the given session of congress covered by the lobbying activity (see variable 'which_congress'); 0 otherwise | |
| terms_consistent_with_ccs | 1 if contains terms that are consistent with CCS but are not ALWAYS CCS (e.g., "carbon dioxide" and "storage' [could be 'energy storage' and 'carbon dioxide emissions', which is not CCS, or it could be 'the storage of and transport of carbon dioxide", which would be more consistent with CCS but might not be caught as 'definitely ccs' because of the atypical phrasing); 0 otherwise | |
| terms_could_be_ccs | 1 if activity description contains terms that sometimes (or even usually) mean we're dealing with CCS (e.g., 'underground injection control'); 0 otherwise | |
| ccs_because_of_who_says_it |  | |
