import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Advanced Ranking-Based Conjoint Analyser by <a href='SumanEcon'>SumanEcon",
    unsafe_allow_html=True
)
st.markdown(
    """
    <p align="center">
      <a href="https://github.com/DenverCoder1/readme-typing-svg">
        <img src="https://readme-typing-svg.herokuapp.com?font=Time+New+Roman&color=yellow&size=30&center=true&vCenter=true&width=600&height=100&lines=Conjoint+Analysis+Made+Simple!;rankconjoint_analyser-2.0;" alt="Typing SVG">
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
    st.dataframe(df.head())

    # Identify the last column as the ranking column
    ranking_column = df.columns[-1]
    
    # Conjoint attributes and model specification
    conjoint_attributes = df.columns[:-1]  # Assuming the last column is 'ranking'
    model = f'{ranking_column} ~ ' + ' + '.join([f'C({attr}, Sum)' for attr in conjoint_attributes])
    
    # Fit the model
    model_fit = smf.ols(model, data=df).fit()
    st.write("### Model Summary:")
    st.write(model_fit.summary())

    # Advanced statistical metrics
    st.write("### Advanced Model Metrics:")
    st.write(f"Adjusted R-squared: {model_fit.rsquared_adj:.4f}")
    st.write(f"AIC (Akaike Information Criterion): {model_fit.aic:.4f}")
    st.write(f"BIC (Bayesian Information Criterion): {model_fit.bic:.4f}")

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
        new_part_worth.append((-1) * sum(new_part_worth))  # Constraint part-worths sum to 0
        important_levels[item] = np.argmax(new_part_worth)
        part_worth.append(new_part_worth)
        part_worth_range.append(max(new_part_worth) - min(new_part_worth))

    # Attribute importance
    attribute_importance = [round(100 * (i / sum(part_worth_range)), 2) for i in part_worth_range]
    
    # Creating a single DataFrame to hold all the attribute, part-worth, and importance information
    part_worth_data = []
    for item, pw, levels, importance in zip(conjoint_attributes, part_worth, level_name, attribute_importance):
        for level, value in zip(levels, pw):
            part_worth_data.append({
                'Attribute': item,
                'Level': level,
                'Part-Worth': round(value, 4),
                'Relative Importance (%)': importance
            })
    
    # Convert to DataFrame
    part_worth_df = pd.DataFrame(part_worth_data)
    
    # Display the table in Streamlit
    st.write("### Part-Worths and Attribute Importance Table:")
    st.dataframe(part_worth_df)

    # Visualizing attribute importance using Plotly
    fig_importance = px.bar(
        x=conjoint_attributes, 
        y=attribute_importance, 
        labels={'x':'Attributes', 'y':'Importance'},
        title='Relative Importance of Attributes'
    )
    st.plotly_chart(fig_importance)

    # PCA for attribute importance (optional advanced technique)
    st.write("### PCA Analysis on Attributes:")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[conjoint_attributes])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    fig_pca = px.scatter(
        pca_result, 
        x=0, y=1, 
        labels={'0':'PCA 1', '1':'PCA 2'}, 
        title='PCA of Attribute Importance'
    )
    st.plotly_chart(fig_pca)

    # Utility calculation
    utility = []
    for i in range(df.shape[0]):
        score = sum([part_worth_df.loc[(part_worth_df['Attribute'] == attr) & (part_worth_df['Level'] == df[attr][i]), 'Part-Worth'].values[0] for attr in conjoint_attributes])
        utility.append(score)
    df['utility'] = utility

    # Profile with the highest utility score
    best_profile = df.iloc[np.argmax(utility)]
    st.write("### The profile that has the highest utility score:")
    st.write(best_profile)

    # Preferred levels in each attribute
    st.write("### Preferred levels in each attribute:")
    for i, item in enumerate(conjoint_attributes):
        st.write(f"Preferred level in {item} is :: {level_name[i][important_levels[item]]}")

    # Correlation heatmap between attributes
    st.write("### Correlation Heatmap between Attributes:")
    plt.figure(figsize=(10, 5))
    corr = df[conjoint_attributes].apply(lambda x: pd.factorize(x)[0]).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)
