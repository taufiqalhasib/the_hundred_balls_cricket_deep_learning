import streamlit as st
from streamlit_option_menu import option_menu

import pickle
import csv
import pandas as pd
import numpy as np
import math
from math import expm1

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

mlp_score = pickle.load(open('the_hundred_first_innings_score_final_model.pkl', 'rb'))
score_portability_df_male = pd.read_csv(r'the_hundred_first_innings_score.csv')

mlp_first_win = pickle.load(open('the_hundred_first_innings_win_final_model.pkl', 'rb'))
first_innings_win_portability_df = pd.read_csv(r'the_hundred_first_innings_win.csv')

mlp_second_win = pickle.load(open('the_hundred_second_innings_win_final_model.pkl', 'rb'))
second_innings_win_portability_df = pd.read_csv(r'the_hundred_second_innings_win.csv')

teams = ['Oval Invincibles',
         'London Spirit',
         'Southern Brave',
         'Welsh Fire',
         'Birmingham Phoenix',
         'Northern Superchargers',
         'Trent Rockets',
         'Manchester Originals']

venues = ['The Kia Oval', 
          'Edgbaston', 
          'Trent Bridge', 
          'Headingley',
          'Emirates Old Trafford', 
          'Sophia Gardens', 
          'Lords',
          'The Ageas Bowl']

gender = ['Male',
          'Female']


st.title("The Hundred")


def streamlit_menu():
        selected = option_menu(
            menu_title=None,  # required
            options=["First Innings Score", "First Innings Win", "Second Innings Win"],  # required
            icons=["book", "book", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected


selected = streamlit_menu()

if selected == "First Innings Score":
    st.title("First Innings Score Probability")

    gender = st.selectbox('Select gender:', sorted(gender))

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('Select batting team:', sorted(teams))
    with col2:
        bowling_team = st.selectbox('Select bowling team:', sorted(teams))

    venue = st.selectbox('Select venue:', sorted(venues))

    col3, col4, col5 = st.columns(3)
    with col3:
        current_score = st.number_input('Current Score:')
    with col4:
        balls = st.number_input('Balls:')
    with col5:
        wickets = st.number_input('Wickets:')

    if st.button('Prediction'):
        balls_left = 100 - balls
        wickets_left = 10 - wickets
        crr = round(((current_score * 5) / balls), 2)
        if str(gender) == 'Male':
            gender = 'male'
        else:
            gender = 'female'

        # Score Prediction
        input_df = pd.DataFrame(
            {'batting_team': [str(batting_team)], 'bowling_team': [str(bowling_team)], 'venue': [str(venue)],
             'gender': [str(gender)], 'current_score': [int(current_score)], 'balls_bowled': [int(balls)],
             'balls_left': [int(balls_left)], 'wickets_left': [int(wickets_left)], 'crr': [crr]})

        transformer = make_column_transformer(
            (OneHotEncoder(sparse=False, drop='first'), [
                'batting_team', 'bowling_team', 'venue', 'gender'
            ]),
            (StandardScaler(), [
                'current_score', 'balls_bowled', 'balls_left', 'wickets_left', 'crr'
            ])
        )
        X = score_portability_df_male.drop(columns=['runs_remaining'], axis=1)
        prediction = mlp_score.predict(transformer.fit(X).transform(input_df))
        predicted_score = math.ceil(prediction.flatten()[0])

        st.subheader(f"Predicted Score: {str(round(predicted_score + current_score) - 8)} to "
                f"{str(round(predicted_score + current_score) + 2)} runs")

if selected == "First Innings Win":
    st.title("First Innings Win Probability")

    gender1 = st.selectbox('Select gender:', sorted(gender))

    col1, col2 = st.columns(2)
    with col1:
        batting_team1 = st.selectbox('Select batting team:', sorted(teams))
    with col2:
        bowling_team1 = st.selectbox('Select bowling team:', sorted(teams))

    venue1 = st.selectbox('Select venue:', sorted(venues))
    total1 = st.number_input('Total Score:')

    if st.button('Prediction'):
        if str(gender1) == 'Male':
            gender1 = 'male'
        else:
            gender1 = 'female'

        transformer1 = make_column_transformer(
                    (OneHotEncoder(sparse=False, drop='first'), [
                        'batting_team', 'bowling_team', 'venue', 'gender'
                    ]),
                    (MinMaxScaler(), [
                        'runs_y'
                    ])
                )
        X1 = first_innings_win_portability_df.drop(columns=['result_first_innings'])
        input_df1 = pd.DataFrame(
            {'batting_team': [str(batting_team1)], 'bowling_team': [str(bowling_team1)], 'venue': [str(venue1)],
             'gender': [str(gender1)], 'runs_y': [int(round(total1))]})

        result = mlp_first_win.predict_proba(transformer1.fit(X1).transform(input_df1))

        loss1 = result[0][0]
        win1 = result[0][1]

        # Score prediction display
        st.subheader(f"Win Probability: {str(batting_team1)} - {str(round(result[0][1] * 100))}% : "
                f"{str(round(result[0][0] * 100))}% - {str(bowling_team1)}")


if selected == "Second Innings Win":
    st.title("Second Innings Win Probability")

    gender2 = st.selectbox('Select gender:', sorted(gender))

    col1, col2 = st.columns(2)
    with col1:
        batting_team2 = st.selectbox('Select batting team:', sorted(teams))
    with col2:
        bowling_team2 = st.selectbox('Select bowling team:', sorted(teams))

    venue2 = st.selectbox('Select venue:', sorted(venues))

    col3, col4, col5, col6 = st.columns(4)
    with col3:
        runs_left2 = st.number_input('Runs Left:')
    with col4:
        balls_left2 = st.number_input('Balls Left:')
    with col5:
        wickets_left2 = st.number_input('Wickets Left:')
    with col6:
        target2 = st.number_input('Target Score:')

    if st.button('Prediction'):
        balls2 = 100 - balls_left2
        current_score2 = target2 - runs_left2
        crr2 = round(((current_score2 * 5) / balls2), 2)
        rrr2 = round(((runs_left2 * 5) / balls_left2), 2)
        if str(gender2) == 'Male':
            gender2 = 'male'
        else:
            gender2 = 'female'

        # Score Prediction
        input_df2 = pd.DataFrame(
            {'batting_team': [str(batting_team2)], 'bowling_team': [str(bowling_team2)], 'venue': [str(venue2)],
             'gender': [str(gender2)], 'runs_left': [int(runs_left2)], 'balls_left': [int(balls_left2)],
             'wickets_left': [int(wickets_left2)], 'target_runs': [int(target2)], 'crr': [float(crr2)],
             'rrr': [float(rrr2)]})

        transformer2 = make_column_transformer(
            (OneHotEncoder(sparse=False, drop='first'), [
                'batting_team', 'bowling_team', 'venue', 'gender'
            ]),
            (MinMaxScaler(), [
                'runs_left', 'balls_left', 'wickets_left', 'target_runs', 'crr', 'rrr'
            ])
        )
        X2 = second_innings_win_portability_df.drop(columns=['result'])
        result2 = mlp_second_win.predict_proba(transformer2.fit(X2).transform(input_df2))

        loss2 = result2[0][0]
        win2 = result2[0][1]

        # Score prediction display
        st.subheader(f"Win Probability: {str(batting_team2)} - {str(round(win2 * 100))}% : {str(round(loss2 * 100))}% -"
                f" {str(bowling_team2)}")
