import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration and title
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Conjoint Analysis App by <a href='SumanEcon'>SumanEcon</a></h1>",
    unsafe_allow_html=True
)

# App note on conjoint methods
st.markdown("""
### Conjoint Analysis Methods:
1. **Ranking-Based Conjoint Analysis:** Used when respondents rank profiles based on preference.
2. **Choice-Based Conjoint Analysis (CBC):** Used when respondents choose one profile from multiple options.
3. **Rating-Based Conjoint Analysis:** Used when respondents rate profiles on a scale (e.g., 1-10).
**Note:** The app automatically selects the method based on the structure of your uploaded dataset.
""")

# Upload data
uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=["csv", "xlsx", "xls"])
if uploaded_file is not None:
    # Read the uploaded data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("### Data Preview:")
    st.write(df.head())

    # Check for ranking, choice, or rating columns to determine which method is applicable
    last_column = df.columns[-1].lower()

    if 'rank' in last_column:  # Ranking-Based Conjoint Analysis
        st.write("### Ranking-Based Conjoint Analysis Selected")
        
        conjoint_attributes = df.columns[:-1]  # Assuming the last column is ranking
        ranking_column = df.columns[-1]
        
        # Build model
        model = f'{ranking_column} ~ ' + ' + '.join([f'C({attr}, Sum)' for attr in conjoint_attributes])
        model_fit = smf.ols(model, data=df).fit()
        
        st.write("### Model Summary:")
        st.write(model_fit.summary())
        
        # Interpretation
        st.markdown("""
        **Interpretation:**
        - **Coefficients:** Higher values indicate a higher preference for that level.
        - **R-squared:** Higher values suggest better model fit.
        """)

        # Extracting part-worths and importance
        level_name = []
        part_worth = []
        part_worth_range = []
        important_levels = {}
        end = 1

        for item in conjoint_attributes:
            nlevels = len(list(np.unique(df[item])))
            level_name.append(list(np.unique(df[item])))

            begin = end
            end = begin + nlevels - 1

            new_part_worth = list(model_fit.params[begin:end])
            new_part_worth.append((-1) * sum(new_part_worth))  # Ensure the part-worths sum to zero
            important_levels[item] = np.argmax(new_part_worth)
            part_worth.append(new_part_worth)
            part_worth_range.append(max(new_part_worth) - min(new_part_worth))

        # Attribute importance
        attribute_importance = [round(100 * (i / sum(part_worth_range)), 2) for i in part_worth_range]
        
        st.write("### Attribute Importance:")
        importance_dict = dict(zip(conjoint_attributes, attribute_importance))
        st.write(importance_dict)

        # Plot part-worth utilities for each attribute
        for i, item in enumerate(conjoint_attributes):
            plt.figure(figsize=(10, 5))
            sns.barplot(x=level_name[i], y=part_worth[i])
            plt.title(f'Part-Worth Utilities for {item}')
            plt.xlabel(f'{item} Levels')
            plt.ylabel('Part-Worth Utility')
            st.pyplot(plt)
            st.write(f"**Interpretation:** Higher part-worth utilities indicate stronger preferences for the corresponding level of {item}.")

        # Plot for relative importance
        plt.figure(figsize=(10, 5))
        sns.barplot(x=conjoint_attributes, y=attribute_importance)
        plt.title('Relative Importance of Attributes')
        plt.xlabel('Attributes')
        plt.ylabel('Importance (%)')
        st.pyplot(plt)
        st.write("**Interpretation:** Higher importance percentages indicate more influential attributes in decision-making.")

    elif 'choice' in last_column:  # Choice-Based Conjoint Analysis
        st.write("### Choice-Based Conjoint Analysis Selected")
        
        # Assuming the data has choice indicators (1/0) in the last column
        conjoint_attributes = df.columns[:-1]
        choice_column = df.columns[-1]

        # Logistic regression model
        model = f'{choice_column} ~ ' + ' + '.join([f'C({attr}, Sum)' for attr in conjoint_attributes])
        model_fit = smf.logit(model, data=df).fit()
        
        st.write("### Model Summary:")
        st.write(model_fit.summary())
        
        # Interpretation
        st.markdown("""
        **Interpretation:**
        - **Coefficients:** Show how likely a respondent is to choose a profile based on its attributes.
        - **p-values:** Significance of each attribute.
        - **Odds Ratios:** Exp of coefficients can be interpreted as odds ratios.
        """)

        # Additional plots for choice-based conjoint
        # Plot for attribute importance (using odds ratios)
        odds_ratios = np.exp(model_fit.params)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=odds_ratios.index, y=odds_ratios)
        plt.title('Odds Ratios of Attributes')
        plt.xlabel('Attributes')
        plt.ylabel('Odds Ratio')
        st.pyplot(plt)
        st.write("**Interpretation:** Odds ratios show the change in odds of choosing a profile as an attribute level changes. Higher odds ratios indicate stronger influence.")

    elif 'rate' in last_column or df[last_column].dtype in ['int64', 'float64']:  # Rating-Based Conjoint Analysis
        st.write("### Rating-Based Conjoint Analysis Selected")
        
        # Assuming the last column is a rating scale (e.g., 1-10)
        conjoint_attributes = df.columns[:-1]
        rating_column = df.columns[-1]

        # Linear regression model
        model = f'{rating_column} ~ ' + ' + '.join([f'C({attr}, Sum)' for attr in conjoint_attributes])
        model_fit = smf.ols(model, data=df).fit()
        
        st.write("### Model Summary:")
        st.write(model_fit.summary())
        
        # Interpretation
        st.markdown("""
        **Interpretation:**
        - **Coefficients:** Higher values suggest a stronger impact of the attribute level on the rating.
        - **p-values:** Indicate the statistical significance.
        - **R-squared:** Explains the variance in the rating scale.
        """)

        # Plot the importance of attributes for Rating-Based Conjoint Analysis
        level_name = []
        part_worth = []
        part_worth_range = []
        for item in conjoint_attributes:
            nlevels = len(list(np.unique(df[item])))
            level_name.append(list(np.unique(df[item])))

            # Extract part-worths
            part_worths = model_fit.params.filter(like=item)
            part_worth.append(part_worths)
            part_worth_range.append(max(part_worths) - min(part_worths))
        
        attribute_importance = [round(100 * (i / sum(part_worth_range)), 2) for i in part_worth_range]
        
        # Plot for part-worth utilities
        for i, item in enumerate(conjoint_attributes):
            plt.figure(figsize=(10, 5))
            sns.barplot(x=level_name[i], y=part_worth[i])
            plt.title(f'Part-Worth Utilities for {item}')
            plt.xlabel(f'{item} Levels')
            plt.ylabel('Part-Worth Utility')
            st.pyplot(plt)
            st.write(f"**Interpretation:** Higher part-worth utilities indicate stronger preferences for the corresponding level of {item}.")

        # Plot for relative importance
        plt.figure(figsize=(10, 5))
        sns.barplot(x=conjoint_attributes, y=attribute_importance)
        plt.title('Relative Importance of Attributes')
        plt.xlabel('Attributes')
        plt.ylabel('Importance (%)')
        st.pyplot(plt)
        st.write("**Interpretation:** Higher importance percentages indicate more influential attributes in decision-making.")
        
    else:
        st.write("### No suitable Conjoint Analysis method could be determined for this dataset.")
else:
    st.write("### Please upload a CSV, XLS, or XLSX file.")
