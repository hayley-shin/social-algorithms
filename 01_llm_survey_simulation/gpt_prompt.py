# Name: Tracy Chen, Hayley Shin
# Group No: 7

import csv
import random
import re
from openai import OpenAI
import time
from tqdm import tqdm
import pandas as pd
import numpy as np

client = OpenAI()

ORIGINAL_COLUMNS = [
    'RespondentID',
    'In your opinion, which sentence is more gramatically correct?',
    'Prior to reading about it above, had you heard of the serial (or Oxford) comma?',
    'How much, if at all, do you care about the use (or lack thereof) of the serial (or Oxford) comma in grammar?',
    'How would you write the following sentence?',
    'When faced with using the word "data", have you ever spent time considering if the word was a singular or plural noun?',
    'How much, if at all, do you care about the debate over the use of the word "data" as a singluar or plural noun?',
    'In your opinion, how important or unimportant is proper use of grammar?',
    'Gender',
    'Age',
    'Household Income',
    'Education',
    'Location (Census Region)'
]

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


def generate_gpt_prompts(data):
    prompts = []
    for row in data:
        age = row['Age']
        gender = row['Gender']
        income = row['Household Income']
        education = row['Education']
        location = row['Location (Census Region)']
        
        prompt = f"""You are participating in a survey about grammar and punctuation preferences.
        You are {age} years old {gender} with a household income of {income}, education level of {education}, and lives in {location}. 
        Please answer the following questions as someone with this demographic profile would. 
        QUESTIONS:

        1. {SUBSTANTIVE_QUESTIONS['Q1']}
        {chr(10).join(SUBSTANTIVE_ANSWER_OPTIONS['Q1'])}

        2. {SUBSTANTIVE_QUESTIONS['Q2']}
        {chr(10).join(SUBSTANTIVE_ANSWER_OPTIONS['Q2'])}

        3. {SUBSTANTIVE_QUESTIONS['Q3']}
        {chr(10).join(SUBSTANTIVE_ANSWER_OPTIONS['Q3'])}
        
        4. {SUBSTANTIVE_QUESTIONS['Q4']}
        {chr(10).join(SUBSTANTIVE_ANSWER_OPTIONS['Q4'])}

        5. {SUBSTANTIVE_QUESTIONS['Q5']}
        {chr(10).join(SUBSTANTIVE_ANSWER_OPTIONS['Q5'])}
        
        6. {SUBSTANTIVE_QUESTIONS['Q6']}
        {chr(10).join(SUBSTANTIVE_ANSWER_OPTIONS['Q6'])}

        7. {SUBSTANTIVE_QUESTIONS['Q7']}
        {chr(10).join(SUBSTANTIVE_ANSWER_OPTIONS['Q7'])}
        
        CRITICAL: Respond in this EXACT format: A,B,C,A,B,D,E
        Use ONLY letters (A-E as appropriate), separated by commas, with NO spaces, NO quotes, NO explanation, and NO other text."""
                
        prompts.append(prompt)
    return prompts

def poll_gpt(gpt_prompts, num_responses, description="Polling GPT"):
    responses = []
    for i in tqdm(range(num_responses), desc=description):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": gpt_prompts[i]}
            ],
            max_tokens=100,
            n=1,
            temperature=1.0,
        )
        responses.append(response.choices[0].message.content.strip())
        time.sleep(0.3)  # Be nice to the API
    return responses

def parse_response(response_text):
    cleaned = response_text.strip().upper()
    pattern = r'([A-E]),([A-E]),([A-E]),([A-E]),([A-E]),([A-E]),([A-E])'
    match = re.search(pattern, cleaned)
    
    if match:
        return list(match.groups())
    
    if ',' in cleaned:
        answers = [a.strip() for a in cleaned.split(',')]
        answers = [a for a in answers if a in 'ABCDE']
        if len(answers) == 7:
            return answers
    
    return [None] * 7

def map_answers_to_text(letter_answers):
    mappings = [
        {
            'A': "It's important for a person to be honest, kind and loyal.",
            'B': "It's important for a person to be honest, kind, and loyal."
        },
        {
            'A': 'Yes',
            'B': 'No'
        },
        {
            'A': 'A lot',
            'B': 'Some',
            'C': 'Not much',
            'D': 'Not at all'
        },
        {
            'A': "Some experts say it's important to drink milk, but the data is inconclusive.",
            'B': "Some experts say it's important to drink milk, but the data are inconclusive."
        },
        {
            'A': 'Yes',
            'B': 'No'
        },
        {
            'A': 'A lot',
            'B': 'Some',
            'C': 'Not much',
            'D': 'Not at all'
        },
        {
            'A': 'Very important',
            'B': 'Somewhat important',
            'C': 'Neither important nor unimportant (neutral)',
            'D': 'Somewhat unimportant',
            'E': 'Very unimportant'
        }
    ]
    
    text_answers = []
    for i, letter in enumerate(letter_answers):
        if letter and i < len(mappings):
            text_answers.append(mappings[i].get(letter, ''))
        else:
            text_answers.append('')
    
    return text_answers

def save_responses_to_csv(output_file, selected_data, parsed_responses, id_prefix):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(ORIGINAL_COLUMNS)
        
        for i, (demo_row, parsed_ans) in enumerate(zip(selected_data, parsed_responses)):
            text_answers = map_answers_to_text(parsed_ans)
            row = [
                f"{id_prefix}_{i+1}",
                text_answers[0], 
                text_answers[1],  
                text_answers[2],  
                text_answers[3],  
                text_answers[4],  
                text_answers[5],  
                text_answers[6],  
                demo_row['Gender'],
                demo_row['Age'],
                demo_row['Household Income'],
                demo_row['Education'],
                demo_row['Location (Census Region)']
            ]
            writer.writerow(row)

def sample_from_baseline(survey_file, num_samples):
    survey_data = []
    with open(survey_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            survey_data.append(row)
    
    random.seed(42)
    selected_data = random.sample(survey_data, num_samples)
    return selected_data

def sample_from_census_weights(census_file, num_samples):
    ### for Step 8: Sample demographics from census data based on population weights
    census_df = pd.read_csv(census_file)
    
    weights = census_df['count'].values
    probabilities = weights / weights.sum()
    
    np.random.seed(42)
    sampled_indices = np.random.choice(
        len(census_df), 
        size=num_samples, 
        replace=True, 
        p=probabilities
    )
    
    sampled_data = []
    for idx in sampled_indices:
        row = census_df.iloc[idx]
        sampled_data.append({
            'Gender': row['sex'],
            'Age': row['age'],
            'Household Income': row['income'],
            'Education': row['education'],
            'Location (Census Region)': row['location']
        })
    return sampled_data


def step2_baseline_survey():
    BASELINE_INPUT = '../comma-survey.csv'
    BASELINE_OUTPUT = '../gpt_comma_survey.csv'
    NUM_RESPONSES = 300
    
    selected_data = sample_from_baseline(BASELINE_INPUT, NUM_RESPONSES)
    gpt_prompts = generate_gpt_prompts(selected_data)
    gpt_responses = poll_gpt(gpt_prompts, NUM_RESPONSES, "Step 2: Baseline Demographics")
    
    parsed_responses = []
    successful = 0
    failed = 0
    for response in gpt_responses:
        parsed = parse_response(response)
        parsed_responses.append(parsed)
        if None not in parsed:
            successful += 1
        else:
            failed += 1
    
    print(f"Successfully parsed: {successful}/{NUM_RESPONSES}")
    print(f"Failed to parse: {failed}/{NUM_RESPONSES}")
    
    # Save to CSV
    print(f"\nSaving results to {BASELINE_OUTPUT}...")
    save_responses_to_csv(BASELINE_OUTPUT, selected_data, parsed_responses, "GPT")

def step8_census_survey():
    CENSUS_INPUT = '../census_population_counts.csv'
    CENSUS_OUTPUT = '../gpt_census_survey.csv'
    NUM_RESPONSES = 300

    selected_data = sample_from_census_weights(CENSUS_INPUT, NUM_RESPONSES)
    gpt_prompts = generate_gpt_prompts(selected_data)
    gpt_responses = poll_gpt(gpt_prompts, NUM_RESPONSES, "Step 8: Census Demographics")
    
    parsed_responses = []
    successful = 0
    failed = 0
    for response in gpt_responses:
        parsed = parse_response(response)
        parsed_responses.append(parsed)
        if None not in parsed:
            successful += 1
        else:
            failed += 1
    
    print(f"Successfully parsed: {successful}/{NUM_RESPONSES}")
    print(f"Failed to parse: {failed}/{NUM_RESPONSES}")
    save_responses_to_csv(CENSUS_OUTPUT, selected_data, parsed_responses, "GPT_CENSUS")
    



step2_baseline_survey()
step8_census_survey()