# Name: Tracy Chen, Hayley Shin
# Group No: 7

import pandas as pd
import matplotlib.pyplot as plt
import textwrap

def explore_data(df):
    print("-" * 40)
    print("Basic Information about the dataset:")
    print(f"Data Shape: {df.shape}")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

def missing_data(df):
    print("-" * 40)
    print("Missing Data Analysis:")
    missing_info = df.isnull().sum()
    missing_percent = (missing_info / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_info, 'Percentage': missing_percent.round(2)})
    print(missing_df)

def demographics(df):
    print("-" * 40)
    print("Demographic Distribution:")
    demographic_cols = ['Gender', 'Age', 'Household Income', 'Education', 'Location (Census Region)']
    # for col in demographic_cols:
    #     if col in df.columns:
    #         print(f"\n{col} Distribution:")
    #         counts = df[col].value_counts(dropna=False)
    #         percentages = (counts / len(df)) * 100
    #         result = pd.DataFrame({'Count': counts,'Percentage': percentages.round(2)})
    #         print(result)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(demographic_cols):
        data = df[col].fillna('Missing').value_counts()
        data.plot(kind='barh', ax=axes[idx])
        axes[idx].set_title(f'{col}')
        axes[idx].set_xlabel('Count')
        
    axes[-1].axis('off') 
    
    plt.tight_layout()
    plt.savefig('./demographics.png', dpi=300, bbox_inches='tight')
    # plt.savefig('./img_gpt/demographics.png', dpi=300, bbox_inches='tight') # this is for step 3
    print("\nDemographic plots saved as 'img_gpt/demographics.png'")
    plt.show()

def substantive_questions(df):
    print("-" * 40)
    print("Substantive Questions Analysis:")
    exclude = ['RespondentID', 'Gender', 'Age', 'Household Income', 'Education', 'Location (Census Region)']

    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    plot_idx = 0

    for col in df.columns:
        if col not in exclude:
            print(f"\n{col} Distribution:")
            
            # counts = df[col].value_counts(dropna=False)
            # percentages = (counts / len(df)) * 100
            # result = pd.DataFrame({'Count': counts,'Percentage': percentages.round(2)})
            # print(result)

            data = df[col].fillna('Missing').value_counts().sort_values()
            data.plot(kind='barh', ax=axes[plot_idx])

            # wrap y-axis labels
            labels = [textwrap.fill(label.get_text(), width=30) for label in axes[plot_idx].get_yticklabels()]
            axes[plot_idx].set_yticklabels(labels)

            wrapped_title = textwrap.fill(col, width=60) # only to wrap title
            axes[plot_idx].set_title(wrapped_title)
            axes[plot_idx].set_xlabel('Count')
            axes[plot_idx].set_ylabel("")

            plot_idx += 1

    # hide unused axes
    for j in range(plot_idx, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig('./substantive_questions.png', dpi=300, bbox_inches='tight')
    # plt.savefig('./img_gpt/substantive_questions.png', dpi=300, bbox_inches='tight') # this is for step 3
    plt.show()
    
    
    
df = pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/data/master/comma-survey/comma-survey.csv')
# df = pd.read_csv('../gpt_comma_survey.csv')
explore_data(df)
missing_data(df)    
demographics(df)
substantive_questions(df)