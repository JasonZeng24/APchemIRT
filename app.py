import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import time

# --- 1. CORE IRT FUNCTIONS (No changes needed here) ---

def sigmoid(x):
    """The logistic function."""
    x = np.clip(x, -100, 100) # Clip to avoid overflow
    return 1 / (1 + np.exp(-x))

def prob_correct(theta, a, b):
    """Calculates the probability of a correct answer using the 2-PL IRT model."""
    return sigmoid(a * (theta - b))

def fisher_information(theta, a, b):
    """Calculates the Fisher Information for a given item."""
    p = prob_correct(theta, a, b)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (a**2) * p * (1 - p)

def negative_log_likelihood(theta, response_history_df):
    """Negative log-likelihood function for MLE ability estimation."""
    a_params = response_history_df['a']
    b_params = response_history_df['b']
    responses = response_history_df['correct']
    p_correct = prob_correct(theta, a_params, b_params)
    likelihoods = np.where(responses == 1, p_correct, 1 - p_correct)
    likelihoods = np.clip(likelihoods, 1e-9, 1.0)
    log_likelihood = np.sum(np.log(likelihoods))
    return -log_likelihood

# --- 2. DATA LOADING AND APP HELPERS ---

@st.cache_data
def load_question_bank(file_path):
    """
    MODIFIED: This version no longer calls st.toast or st.error.
    Instead, it returns a message to be displayed by the main app.
    """
    try:
        df = pd.read_csv(file_path)
        df.rename(columns={'initial_discrimination_a': 'a', 'initial_difficulty_b': 'b'}, inplace=True)
        df['id'] = df.index
        option_cols = ['option_a', 'option_b', 'option_c', 'option_d']
        df['options'] = df[option_cols].apply(lambda x: [i for i in x if pd.notna(i)], axis=1)

        def get_correct_answer(row):
            correct_letter = row['correct_answer']
            if correct_letter == 'A': return row['option_a']
            if correct_letter == 'B': return row['option_b']
            if correct_letter == 'C': return row['option_c']
            if correct_letter == 'D': return row['option_d']
            return None
        df['answer'] = df.apply(get_correct_answer, axis=1)

        original_count = len(df)
        df = df[df['options'].str.len() > 0].copy()
        questions_removed = original_count - len(df)

        message = f"Removed {questions_removed} question(s) with no answer options." if questions_removed > 0 else None

        # Return the dataframe and a potential message
        return df, message

    except Exception as e:
        # Return None for the dataframe and the error message
        return None, f"An error occurred while loading or processing the CSV: {e}"
def select_next_question(theta_est, available_questions_df):
    """
    MODIFIED: This version is more robust and avoids KeyError.
    It finds the integer position of the best question instead of its index label.
    """
    if available_questions_df.empty:
        return None
    
    # Calculate Fisher Information for all available questions
    infos = fisher_information(theta_est, available_questions_df['a'], available_questions_df['b'])
    
    # Convert to a NumPy array to safely find the position of the max value
    infos_np = infos.to_numpy()
    
    # Find the integer position (e.g., 0, 1, 2...) of the question with the highest information
    best_question_position = np.argmax(infos_np)
    
    # Retrieve the question using its integer position with .iloc
    return available_questions_df.iloc[best_question_position]

def update_ability_estimate(response_history_df):
    """Updates the student's ability estimate using Maximum Likelihood Estimation (MLE)."""
    if len(response_history_df) < 2 or response_history_df['correct'].nunique() < 2:
        last_response_correct = response_history_df['correct'].iloc[-1]
        current_theta = st.session_state.theta_est
        return current_theta + 0.25 if last_response_correct else current_theta - 0.25

    initial_guess = st.session_state.theta_est
    result = minimize(negative_log_likelihood, x0=initial_guess, args=(response_history_df,), method='BFGS')
    return np.clip(result.x[0], -4, 4)

# --- 3. STREAMLIT APP INITIALIZATION AND LAYOUT ---

# Call the function and unpack the returned tuple
question_bank_df, load_message = load_question_bank('questions_expanded.csv')

# Handle any warning or error messages outside the cached function
if load_message:
    # If the dataframe is None, it was an error
    if question_bank_df is None:
        st.error(load_message)
        st.info("Ensure your CSV has columns: 'prose', 'formula', 'option_a', 'option_b', etc.")
    # Otherwise, it was just a warning toast
    else:
        st.toast(load_message, icon="⚠️")

if 'initialized' not in st.session_state and question_bank_df is not None:
    st.session_state.initialized = True
    st.session_state.theta_est = 0.0
    st.session_state.response_history = []
    st.session_state.current_question_id = None
    st.session_state.quiz_finished = False

st.title("🧪 IRT-Powered Adaptive Quiz")
st.markdown("This demo uses an adaptive algorithm to select questions based on your estimated ability level.")

if question_bank_df is None:
    st.warning("Cannot start quiz because the question bank failed to load or process.")
    st.stop()

if st.session_state.quiz_finished:
    st.success(f"🎉 Quiz Complete! Your final ability estimate is: **{st.session_state.theta_est:.2f}**")
    if st.button("Restart Quiz"):
        st.session_state.clear()
        st.rerun()
    st.stop()

answered_ids = [r['id'] for r in st.session_state.response_history]
available_questions = question_bank_df[~question_bank_df['id'].isin(answered_ids)].reset_index(drop=True)

if available_questions.empty and st.session_state.initialized:
    st.session_state.quiz_finished = True
    st.rerun()

current_question = select_next_question(st.session_state.theta_est, available_questions)

if current_question is not None:
    st.session_state.current_question_id = current_question.id
    
    st.header(f"Question {len(answered_ids) + 1} of {len(question_bank_df)}")
    
    # --- NEW DISPLAY LOGIC ---
    # Display the prose part of the question
    if pd.notna(current_question['prose']):
        st.markdown(f"#### {current_question['prose']}")
    
    # Display the formula using the dedicated st.latex function
    if pd.notna(current_question['formula']):
        st.latex(current_question['formula'])
    # --- END OF NEW DISPLAY LOGIC ---

    cols = st.columns(len(current_question['options']))
    for i, option in enumerate(current_question['options']):
        if cols[i].button(option, key=f"q{current_question.id}_opt{i}"):
            is_correct = (option == current_question['answer'])
            response_record = {'id': current_question.id, 'a': current_question.a, 'b': current_question.b, 'correct': 1 if is_correct else 0}
            st.session_state.response_history.append(response_record)
            
            if is_correct: st.success("Correct!")
            else: st.error(f"Incorrect. The correct answer was: {current_question['answer']}")
            
            old_theta = st.session_state.theta_est
            history_df = pd.DataFrame(st.session_state.response_history)
            st.session_state.theta_est = update_ability_estimate(history_df)

            with st.expander("See how your ability estimate changed"):
                st.metric(label="Ability Estimate (θ)", value=f"{st.session_state.theta_est:.2f}", delta=f"{st.session_state.theta_est - old_theta:.2f}")

            time.sleep(1.5)
            st.rerun()

with st.sidebar:
    st.header("⚙️ Adaptive Engine Status")
    st.metric(label="Current Ability Estimate (θ)", value=f"{st.session_state.get('theta_est', 0.0):.2f}")
    st.markdown("---")
    st.subheader("Response History")
    
    # Check if the response history exists and is not empty
    if st.session_state.get('response_history'):
        # Create the DataFrame from the list of response dictionaries
        history_df = pd.DataFrame(st.session_state.response_history)
        
        # Define the columns we would ideally like to show
        cols_to_show = ['id', 'correct', 'b']
        
        # NEW: Check which of our desired columns are actually available in the DataFrame
        available_cols = [col for col in cols_to_show if col in history_df.columns]
        
        # Only proceed if there are columns to display
        if available_cols:
            history_display_df = history_df[available_cols]
            
            # Rename 'b' to 'difficulty' if that column exists
            if 'b' in history_display_df.columns:
                history_display_df = history_display_df.rename(columns={'b': 'difficulty'})

            st.dataframe(history_display_df, use_container_width=True)
        else:
            st.write("History available but columns could not be displayed.")
    else:
        st.write("No responses yet.")