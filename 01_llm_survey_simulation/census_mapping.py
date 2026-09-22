# Name: Tracy Chen, Hayley Shin
# Group No: 7

import json
import pandas as pd
from pathlib import Path


# manually mapping dictionaries
sex_mapping = {"1": "Male","2": "Female"}
age_mapping = {"1": "18-29","2": "30-44","3": "45-60","4": "> 60"}
education_mapping = {
    "1": "Less than high school degree","2": "High school degree",
    "3": "Some college or Associate degree",
    "4": "Bachelor degree", "5": "Graduate degree"
}
income_mapping = {
    "1": "$0 - $24,999", "2": "$25,000 - $49,999",
    "3": "$50,000 - $99,999", "4": "$100,000 - $149,999", "5": "$150,000+"
}
division_mapping = {
    "1": "New England", "2": "Middle Atlantic", "3": "East North Central",
    "4": "West North Central", "5": "South Atlantic", "6": "East South Central",
    "7": "West South Central", "8": "Mountain", "9": "Pacific"
}
    
def load_census_counts(json_file):

    with open(json_file, 'r') as f:
        census_data = json.load(f)
    
    columns = census_data[0] # [["tabulate","SEX", "AGEP_RC1", "SCHL_RC1", "HINCP_RC1", "ucgid"],
    data_rows = census_data[1:]
    census_df = pd.DataFrame(data = data_rows, columns = columns)
    
    # "0300000US1" → "1"
    census_df['division_code'] = census_df['ucgid'].str.extract(r'US(\d+)$')[0]
    
    print("Applying mappings...")
    census_df['sex'] = census_df['SEX'].map(sex_mapping)
    census_df['age'] = census_df['AGEP_RC1'].map(age_mapping)
    census_df['education'] = census_df['SCHL_RC2'].map(education_mapping)
    census_df['income'] = census_df['HINCP_RC1'].map(income_mapping)
    census_df['location'] = census_df['division_code'].map(division_mapping)
    census_df['count'] = pd.to_numeric(census_df['tabulate'])
    print("Mappings applied.")
   
    census_df_clean = census_df.dropna(subset=['sex', 'age', 'education', 'income', 'location'])
    census_counts = census_df_clean[['sex', 'age', 'education', 'income', 'location', 'count']].copy()
    census_counts.to_csv(Path(__file__).resolve().parent / '../census_population_counts.csv', index=False)
    print("Saved to census_population_counts.csv")
    
    return census_counts


BASE = Path(__file__).resolve().parent   # get the path of the folder
census_counts = load_census_counts(BASE / '../Step6_census_count.json')

print("\nFirst 10 rows of census counts:")
print(census_counts.head(10))
