import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration and title
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Ranking-Based Conjoint Analyser by <a href='SumanEcon'>SumanEcon</a></h1>",
    unsafe_allow_html=True
)
st.markdown(
    """
    <p align="center">
      <a href="https://github.com/DenverCoder1/readme-typing-svg">
        <img src="https://readme-typing-svg.herokuapp.com?font=Time+New+Roman&color=yellow&size=30&center=true&vCenter=true&width=600&height=100&lines=Conjoint+Analysis+Made+Simple!;rankconjoint_analyser-1.0;" alt="Typing SVG">
      </a>
    </p>
    """,
    unsafe_allow_html=True
)

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
    
    # Identify the last column as the ranking column
    ranking_column = df.columns[-1]
    
    # Conjoint attributes and model specification
    conjoint_attributes = df.columns[:-1]  # Assuming the last column is 'ranking'
    model = f'{ranking_column} ~ ' + ' + '.join([f'C({attr}, Sum)' for attr in conjoint_attributes])
    
    # Fit the model
    model_fit = smf.ols(model, data=df).fit()
    
    # Display model summary
    st.write("### Model Summary:")
    st.write(model_fit.summary())

    # Interpretation of model summary:
    st.markdown("""
    **Interpretation of the Model Summary:**
    - **Coefficients:** The coefficients represent the part-worth utilities for the different levels of each attribute. A higher coefficient indicates a higher preference for that level.
    - **p-values:** This shows the statistical significance of each level. A p-value < 0.05 suggests that the level has a significant impact on the ranking.
    - **R-squared:** This value tells us how well the model explains the variation in the ranking. Higher values indicate a better fit.
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

    # Interpretation of attribute importance:
    st.markdown("""
    **Interpretation of Attribute Importance:**
    - The relative importance values indicate how much each attribute influences the ranking decision.
    - Attributes with higher values are more important to the respondents, while those with lower values have less influence on their decisions.
    """)

    # Part-worths for each attribute level
    part_worth_dict = {}
    for item, pw, levels in zip(conjoint_attributes, part_worth, level_name):
        st.write(f"### Attribute: {item}")
        st.write(f"    **Relative importance:** {attribute_importance[conjoint_attributes.tolist().index(item)]}%")
        st.write(f"    **Level wise part worths:**")
        for level, value in zip(levels, pw):
            st.write(f"        {level}: {value}")
            part_worth_dict[level] = value

    # Plotting relative importance of attributes
    plt.figure(figsize=(10, 5))
    sns.barplot(x=conjoint_attributes, y=attribute_importance)
    plt.title('Relative Importance of Attributes')
    plt.xlabel('Attributes')
    plt.ylabel('Importance (%)')
    st.pyplot(plt)

    # Utility calculation
    utility = []
    for i in range(df.shape[0]):
        score = sum([part_worth_dict[df[attr][i]] for attr in conjoint_attributes])
        utility.append(score)
    df['utility'] = utility

    # Profile with the highest utility score
    best_profile = df.iloc[np.argmax(utility)]
    st.write("### The profile that has the highest utility score:")
    st.write(best_profile)

    # Interpretation of best profile:
    st.markdown("""
    **Interpretation of the Best Profile:**
    - The profile with the highest utility score represents the combination of attribute levels that is most preferred by respondents.
    - This profile can guide decision-makers to design a product or service with attributes that maximize consumer satisfaction.
    """)

    # Preferred levels in each attribute
    st.write("### Preferred levels in each attribute:")
    for i, item in enumerate(conjoint_attributes):
        st.write(f"Preferred level in **{item}** is: **{level_name[i][important_levels[item]]}**")

    # Interpretation of preferred levels:
    st.markdown("""
    **Interpretation of Preferred Levels:**
    - For each attribute, the level with the highest part-worth utility indicates the respondents' preferred option.
    - These preferred levels are the most favored by respondents and can be prioritized in product development or marketing strategies.
    """)

