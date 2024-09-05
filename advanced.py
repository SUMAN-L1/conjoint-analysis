import streamlit as st
import pandas as pd
import numpy as np
import itertools

# Function to generate profiles and choice sets
def generate_profiles_and_choice_sets():
    # Define attributes and their levels
    brands = ['Samsung', 'Apple', 'Google', 'OnePlus']
    screen_sizes = ['5.5 inches', '6.1 inches', '6.7 inches']
    battery_lives = ['12 hours', '24 hours', '36 hours']
    prices = ['$300', '$500', '$700']
    camera_qualities = ['12 MP', '48 MP', '108 MP']
    
    # Generate all possible profiles
    profiles = list(itertools.product(brands, screen_sizes, battery_lives, prices, camera_qualities))
    profile_df = pd.DataFrame(profiles, columns=['Brand', 'Screen Size', 'Battery Life', 'Price', 'Camera Quality'])
    profile_df['Profile ID'] = profile_df.index + 1

    # Create choice sets (example with 3 options per set)
    num_choice_sets = len(profile_df) // 3
    choice_sets = []
    for i in range(num_choice_sets):
        choice_sets.append(profile_df.iloc[i*3:(i+1)*3])
    
    return profile_df, choice_sets

# Function to simulate respondent choices
def simulate_responses(profile_df, choice_sets):
    responses = []
    for choice_set in choice_sets:
        chosen_option = np.random.choice(choice_set['Profile ID'])
        for _, row in choice_set.iterrows():
            responses.append({
                'Profile ID': row['Profile ID'],
                'Chosen': int(row['Profile ID'] == chosen_option)
            })
    return pd.DataFrame(responses)

# Streamlit app
st.title('Choice-Based Conjoint Analysis')

# Generate profiles and choice sets
profile_df, choice_sets = generate_profiles_and_choice_sets()

# Display profiles and choice sets
st.subheader('Profiles')
st.write(profile_df)

st.subheader('Choice Sets')
for idx, choice_set in enumerate(choice_sets):
    st.write(f'Choice Set {idx + 1}')
    st.write(choice_set)

# Simulate responses
st.subheader('Simulated Responses')
responses_df = simulate_responses(profile_df, choice_sets)
st.write(responses_df)

# Save to CSV
st.download_button(
    label="Download Response Data as CSV",
    data=responses_df.to_csv(index=False),
    file_name='simulated_responses.csv',
    mime='text/csv'
)

# Optional: Add analysis or visualization of the simulated data here
# For example, you could fit a model and visualize results
