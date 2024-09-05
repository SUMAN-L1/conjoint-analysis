import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

# Set up the Streamlit page
st.title('Conjoint Analysis Tool')

# File upload section
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV, XLSX, or XLS", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    # Determine the file type and read the dataset
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Display dataset preview
    st.write("Dataset Preview:")
    st.write(df.head())
    
    # Select the target (dependent) variable
    target_var = st.selectbox('Select the dependent variable (e.g., selected or rating):', df.columns)
    
    # Select columns to exclude from independent variables
    exclude_vars = st.multiselect('Exclude columns (e.g., IDs, non-numeric):', df.columns, default=[target_var])
    
    # Define the independent variables
    independent_vars = [col for col in df.columns if col not in exclude_vars]
    
    # Handle missing values
    df = df.dropna(subset=[target_var])

    y = df[target_var]
    x = df[independent_vars]
    
    # Convert categorical variables into dummy/indicator variables
    xdum = pd.get_dummies(x)
    
    # Regression model
    res = sm.OLS(y, xdum).fit()

    # Show regression summary
    st.write("Regression Summary:")
    st.write(res.summary())

    # Interpretation of regression results
    st.markdown("**Interpretation:**")
    st.write("""
    The regression summary shows the coefficient values for each variable, the p-values indicating the significance
    of the variables, and the R-squared value that tells how well the model fits the data. 
    Coefficients closer to zero indicate a smaller effect, while larger coefficients suggest a stronger influence.
    If the p-value is less than 0.05, the variable is considered significant at a 95% confidence level.
    """)

    # DataFrame for regression results
    df_res = pd.DataFrame({
        'param_name': res.params.keys(),
        'param_w': res.params.values,
        'pval': res.pvalues
    })
    
    # Calculate absolute values and significance at 95% confidence
    df_res['abs_param_w'] = np.abs(df_res['param_w'])
    df_res['is_sig_95'] = df_res['pval'] < 0.05
    df_res['c'] = ['blue' if x else 'red' for x in df_res['is_sig_95']]
    
    # Sort by absolute parameter values
    df_res = df_res.sort_values(by='abs_param_w', ascending=True)

    # Bar plot of parameter weights (Part Worth)
    st.write("Part Worths of Attributes:")
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.title('Part Worth')
    plt.barh(df_res['param_name'], df_res['param_w'], color=df_res['c'])
    plt.xlabel('Parameter Weight')
    st.pyplot(fig)

    # Interpretation of part-worth
    st.markdown("**Interpretation of Part Worths:**")
    st.write("""
    Part worths represent the contribution of each attribute level to the overall preference.
    Blue bars represent significant attributes (p < 0.05), while red bars are not significant. 
    Larger absolute part worths indicate greater importance in the decision-making process.
    """)

    # Compute importance per feature
    range_per_feature = dict()
    for key, coeff in res.params.items():
        feature = key.split('_')[0]
        if feature not in range_per_feature:
            range_per_feature[feature] = []
        range_per_feature[feature].append(coeff)
    
    # Importance calculation
    importance_per_feature = {k: max(v) - min(v) for k, v in range_per_feature.items()}
    total_feature_importance = sum(importance_per_feature.values())
    relative_importance_per_feature = {k: 100 * round(v / total_feature_importance, 3) for k, v in importance_per_feature.items()}

    # Plot of feature importance
    st.write("Feature Importance:")
    alt_data = pd.DataFrame(list(importance_per_feature.items()), columns=['attr', 'importance']).sort_values(by='importance', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.barh(alt_data['attr'], alt_data['importance'])
    plt.xlabel('% importance')
    plt.ylabel('Attributes')
    st.pyplot(fig)

    # Interpretation of feature importance
    st.markdown("**Interpretation of Feature Importance:**")
    st.write("""
    The feature importance plot shows the range of part-worths for each attribute, which represents its overall importance.
    Attributes with larger ranges are more important in influencing decisions.
    """)

    # Plot of relative importance
    st.write("Relative Feature Importance:")
    alt_data = pd.DataFrame(list(relative_importance_per_feature.items()), columns=['attr', 'relative_importance (pct)']).sort_values(by='relative_importance (pct)', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.barh(alt_data['attr'], alt_data['relative_importance (pct)'])
    plt.xlabel('% relative importance')
    plt.ylabel('Attributes')
    st.pyplot(fig)

    # Interpretation of relative importance
    st.markdown("**Interpretation of Relative Importance:**")
    st.write("""
    This plot shows the normalized importance of each attribute as a percentage of the total. 
    Higher percentages indicate more critical attributes in the overall decision-making process.
    """)

else:
    st.write("Please upload a dataset to proceed.")
