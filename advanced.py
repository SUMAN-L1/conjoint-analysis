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

        # Part-worth utilities, relative importance, and plots (similar to your original code)

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

        # Additional plots for choice-based conjoint (importance, odds ratios, etc.)

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

    else:
        st.write("### No suitable Conjoint Analysis method could be determined for this dataset.")
    
    # General plots (Part-worths, Attribute Importance, etc.)
    # Here are plots applicable to all types (e.g., importance, part-worths)

    # Plot for part-worth utilities
    for item in conjoint_attributes:
        # Assume part-worths are calculated based on the model (example for OLS, could change for logistic)
        st.write(f"### Attribute: {item}")
        levels = df[item].unique()
        part_worths = model_fit.params.filter(like=item)
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=levels, y=part_worths)
        plt.title(f'Part-Worth Utilities for {item}')
        plt.xlabel(f'{item} Levels')
        plt.ylabel('Part-Worth Utility')
        st.pyplot(plt)
        st.write(f"**Interpretation:** Higher part-worth utilities indicate stronger preferences for the corresponding level of {item}.")
    
    # General plot for relative importance
    # Assume importance is calculated as in the original code
    relative_importance = []  # Add logic to calculate relative importance
    plt.figure(figsize=(10, 5))
    sns.barplot(x=conjoint_attributes, y=relative_importance)
    plt.title('Relative Importance of Attributes')
    plt.xlabel('Attributes')
    plt.ylabel('Importance (%)')
    st.pyplot(plt)
    st.write("**Interpretation:** Higher importance percentages indicate more influential attributes in decision-making.")

else:
    st.write("### Please upload a CSV, XLS, or XLSX file.")
