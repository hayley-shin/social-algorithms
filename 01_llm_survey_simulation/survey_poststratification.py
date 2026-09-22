# Name: Tracy Chen, Hayley Shin
# Group No: 7

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
from pathlib import Path
import itertools

# Define the columns
# Substantive question columns
SUBSTANTIVE_QUESTIONS = {
    'Q1': "In your opinion, which sentence is more gramatically correct?",
    'Q2': "Prior to reading about it above, had you heard of the serial (or Oxford) comma?",
    'Q3': "How much, if at all, do you care about the use (or lack thereof) of the serial (or Oxford) comma in grammar?",
    'Q4': "How would you write the following sentence?",
    'Q5': 'When faced with using the word "data", have you ever spent time considering if the word was a singular or plural noun?',
    'Q6': 'How much, if at all, do you care about the debate over the use of the word "data" as a singluar or plural noun?',
    'Q7': "In your opinion, how important or unimportant is proper use of grammar?"
}

SUBSTANTIVE_ANSWER_OPTIONS = {
    'Q1': [
        "A) It's important for a person to be honest, kind and loyal.",
        "B) It's important for a person to be honest, kind, and loyal."
    ],
    'Q2': [
        "A) Yes",
        "B) No"
    ],
    'Q3': [
        "A) A lot",
        "B) Some",
        "C) Not much",
        "D) Not at all"
    ],
    'Q4': [
        "A) Some experts say it's important to drink milk, but the data is inconclusive.",
        "B) Some experts say it's important to drink milk, but the data are inconclusive."
    ],
    'Q5': [
        "A) Yes",
        "B) No"
    ],
    'Q6': [
        "A) A lot",
        "B) Some",
        "C) Not much",
        "D) Not at all"
    ],
    'Q7': [
        "A) Very important",
        "B) Somewhat important",
        "C) Neither important nor unimportant (neutral)",
        "D) Somewhat unimportant",
        "E) Very unimportant"
    ]
}

# Demographic columns
DEMOGRAPHIC_COLS = [
    'Gender', 
    'Age', 
    'Household Income', 
    'Education', 
    'Location (Census Region)']

# Survey column names to census column names
CENSUS_SURVEY_MAPPING = {
    'sex': 'Gender',
    'age': 'Age', 
    'income': 'Household Income',
    'education': 'Education', 
    'location': 'Location (Census Region)'
}

# Load csv file and drop NA
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df_dropna = df.dropna(subset=DEMOGRAPHIC_COLS)
    return df, df_dropna

# Preprocess (encoding categorial variables) the X (demographic) variables
def encode_demographics(df_dropna):
    # Create OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Transform the demographic columns
    X_encoded = encoder.fit_transform(df_dropna[DEMOGRAPHIC_COLS])

    # Get feature names (just for the reference)
    feature_names = encoder.get_feature_names_out(DEMOGRAPHIC_COLS)

    return X_encoded, encoder, feature_names

# Train multinomial logistic regression for one question
def train_multinomial_logistic_regression(question, X_encoded, df_dropna):
    
    # Only use rows that has response for this quesiton
    idx = df_dropna[question].notna()

    X = X_encoded[idx]
    y = df_dropna.loc[idx, question]

    # Encode response variable
    encoder_y = LabelEncoder()
    y_encoded = encoder_y.fit_transform(y.astype(str))

    # Check class length
    # if all responses are the same, we should skip model training
    if len(encoder_y.classes_) < 2:
        print(f"\nSkipping '{question}' - because only have 1 response class")
        return None
    
    # Train multinomial logistic regression
    print(f"\nTraining model for : '{question}'")
    lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=5350)
    lr.fit(X, y_encoded)

    # Save and print model information
    model_info = {
        'model': lr,
        'y_encoder': encoder_y,
        'question': question,
        'score': lr.score(X, y_encoded)
    }
    print(f"Model training complete. \nScore: {model_info['score']:.5f}")

    return model_info

# Train multinomial logistic regression for all question (separate model for each question)
def train_all(X_encoded, df_dropna):
    models = {}
    for id, question in SUBSTANTIVE_QUESTIONS.items():
        model_info = train_multinomial_logistic_regression(question, X_encoded, df_dropna)
        if model_info is not None:
            models[id] = model_info
    return models

# Save models
def save_models(models, encoder, feature_names, output_file):
    output = {
        'models': models,
        'encoder': encoder,
        'feature_names': feature_names
    }

    with open(output_file, 'wb') as f:
        pickle.dump(output, f)
    
    print(f"\nResults saved to {output_file}")

# Generate the possible cells of demographic info
def generate_all_cells(encoder, demographic_cols):
    categories = encoder.categories_   # unique categories for each demograpic columns
    all_cells_iter = list(itertools.product(*categories))   # generate all possible cells
    all_cells = []
    for cell in all_cells_iter:   # change to list of dictionaries
        tmp_cell = dict(zip(demographic_cols, cell))
        all_cells.append(tmp_cell)
    return all_cells

# Compute predicted probabilities P(response|cell) for each cell
def compute_cell_estimates(models,encoder, demographic_cells):
    cell_estimates = {}
    cell_df = pd.DataFrame(demographic_cells)
    cell_encoded = encoder.transform(cell_df)

    for id, model_info in models.items():
        
        model = model_info['model']
        y_encoder = model_info['y_encoder']

        # Compute predicted probabilities
        proba = model.predict_proba(cell_encoded)
        
        cell_estimates_df = cell_df.copy()
        response_labels = y_encoder.inverse_transform(model.classes_)   # to get response text
        # Add probability column
        for i, label in enumerate(response_labels):
            cell_estimates_df[label] = proba[:, i]

        cell_estimates[id] = cell_estimates_df

    return cell_estimates

# Merge cell estimates and census population counts
def merge_cell_estimates_and_census(cell_estimates, census_df, demographic_cols):
    census_df = census_df.rename(columns=CENSUS_SURVEY_MAPPING)   # because census column names do not match to survey column names
    merged_dict = {}

    # Merge cell estimates dataframe and census data
    for id, estimates_df in cell_estimates.items():
        merged_df = estimates_df.merge(census_df[demographic_cols + ['count']], on=demographic_cols, how='left')
        merged_dict[id] = merged_df
    
    return merged_dict

# Compute estimates using post-stratification (weighted average)
# Σ(probability * population count) / Σ(population count)
def compute_poststratified_estimates(merged_dict):
    poststratification_dict = {}

    for id, merged_df in merged_dict.items():
        
        # Split response columns
        demographic_count = DEMOGRAPHIC_COLS + ['count']
        response_cols = [col for col in merged_df.columns if col not in demographic_count]

        # Compute weighted average for each response
        count_sum = merged_df['count'].sum()
        estimates = {}
        for res in response_cols:
            estimates[res] = (merged_df[res] * merged_df['count']).sum() / count_sum
        
        poststratification_dict[id] = estimates

    return poststratification_dict

# Compute sample means
def compute_sample_means(df):
    sample_means = {}
    for id, question in SUBSTANTIVE_QUESTIONS.items():
        response_percent = df[question].value_counts(normalize=True)
        sample_means[id] = response_percent.to_dict()
    return sample_means

# Compare sample mean and post-stratified estimate
def compare_estimates(id, question_text, sample_mean, poststratified_estimate):
    print(f"\n{id}: {question_text}")
    print(f"\n{'Response':<75} {'Sample Mean':>14} {'Post-Strat.':>14} {'Difference':>14}")
    
    # Print sample mean, post-stratified estimate and difference
    responses = set(list(sample_mean.keys()) + list(poststratified_estimate.keys()))   # all responses
    for response in sorted(responses):
        sample = sample_mean.get(response, 0.000)
        poststratified = poststratified_estimate.get(response, 0.000)
        difference = poststratified - sample
        print(f"{response:<75} {sample:>14.3%} {poststratified:>14.3%} {difference:>14.3%}")

# Print all comparison of sample mean and post-stratified estimate
def print_all_comparison(sample_means, poststratified_estimates, dataset):
    print(f"\n{dataset}")
    for id in sample_means.keys():
        question_text = SUBSTANTIVE_QUESTIONS[id]
        # if both sample mean and post-stratified estimate exist
        if id in poststratified_estimates:
            compare_estimates(id, question_text, sample_means[id], poststratified_estimates[id])
        # only sample mean exist (all answered same for some questions in GPT data)
        else:
            print(f"\n{id}: {question_text}")
            print(f"\n{'Response':<75} {'Sample Mean':>14} {'Post-Strat.':>14} {'Difference':>14}")
            for response in sorted(sample_means[id].keys()):
                sample = sample_means[id][response]
                print(f"{response:<75} {sample:>14.3%} {'NA':>14} {'NA':>14}")

# Save post-stratified attitudes
def save_results(data_name, sample_means, poststratified_estimates):
    results_rows = []
    for id in sorted(sample_means.keys()):
        if id in poststratified_estimates:
            for response in set(list(sample_means[id].keys()) + list(poststratified_estimates[id].keys())):
                sample = sample_means[id].get(response, 0.000)
                poststratified = poststratified_estimates[id].get(response, 0.000)
                results_rows.append({
                    'Dataset': data_name,
                    'Question_ID': id,
                    'Question': SUBSTANTIVE_QUESTIONS[id],
                    'Response': response,
                    'Sample_Mean': sample,
                    'Post_Stratified_Estimates': poststratified,
                    'Difference': poststratified - sample
                })
    return results_rows


# Get the path of the folder
BASE = Path(__file__).resolve().parent

# Step 4: Post-stratification I on Baseline Data
print(f"\nStep 4: Post-stratification I on Baseline Data")
df, df_dropna = load_data(BASE / '../comma-survey.csv')   # load data
X_encoded, encoder, feature_names = encode_demographics(df_dropna)   # encode X
models = train_all(X_encoded, df_dropna)   # train models
save_models(models, encoder, feature_names, BASE / '../baseline_models.pkl')   # save models

# Step 5: Post-stratification I on LLM Data
print(f"\nStep 5: Post-stratification I on LLM Data")
df_gpt, df_gpt_dropna = load_data(BASE / '../gpt_comma_survey.csv')   # load data
X_gpt_encoded, encoder_gpt, feature_names_gpt = encode_demographics(df_gpt_dropna)   # encode X
models_gpt = train_all(X_gpt_encoded, df_gpt_dropna)   # train models
save_models(models_gpt, encoder_gpt, feature_names_gpt, BASE / '../gpt_models.pkl')   # save models

# Step 7: Post-stratification II - Baseline and GPT Data
print(f"\nStep 7: Post-stratification II - Baseline and GPT Data")
census_df = pd.read_csv(BASE / '../census_population_counts.csv')   # load data
demographic_cells = generate_all_cells(encoder, DEMOGRAPHIC_COLS)   # generate all possible demographic cells

# Post-stratification II - Baseline Data
print(f"\nPost-stratification II - Baseline Data")
cell_estimates_baseline = compute_cell_estimates(models, encoder, demographic_cells)   # compute cell estimates
merged_baseline = merge_cell_estimates_and_census(cell_estimates_baseline, census_df, DEMOGRAPHIC_COLS)   # merge cell estimates and census population counts
poststratified_estimates_baseline = compute_poststratified_estimates(merged_baseline)   # compute estimates using post-stratification
sample_means_baseline = compute_sample_means(df_dropna)   # compute sample means
print_all_comparison(sample_means_baseline, poststratified_estimates_baseline, "BASELINE DATA")   # print results

# Post-stratification II - GPT Data
print(f"\nPost-stratification II - GPT Data")
cell_estimates_gpt = compute_cell_estimates(models_gpt, encoder_gpt, demographic_cells)   # compute cell estimates
merged_gpt = merge_cell_estimates_and_census(cell_estimates_gpt, census_df, DEMOGRAPHIC_COLS)   # merge cell estimates and census population counts
poststratified_estimates_gpt = compute_poststratified_estimates(merged_gpt)   # compute estimates using post-stratification
sample_means_gpt = compute_sample_means(df_gpt_dropna)   # compute sample means
print_all_comparison(sample_means_gpt, poststratified_estimates_gpt, "GPT DATA")   # print results

# Step 8: Validation Survey for Census Population
print(f"\nStep 8: Validation Survey for Census Population")
df_gpt_census = pd.read_csv(BASE / '../gpt_census_survey.csv')
sample_means_gpt_census = compute_sample_means(df_gpt_census)   # compute sample means
print_all_comparison(sample_means_gpt_census, poststratified_estimates_gpt, "GPT DATA - US Census Demographics")   # print results

# Save post-stratified attitudes (to csv file)
results_list = []
results_list.extend(save_results('Baseline', sample_means_baseline, poststratified_estimates_baseline))   # post-stratified attitudes for baseline data
results_list.extend(save_results('GPT', sample_means_gpt, poststratified_estimates_gpt))   # post-stratified attitudes for gpt data
results_df = pd.DataFrame(results_list)
results_df.to_csv(BASE / '../poststratified_attitudes.csv', index=False)   # save results to csv file