#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 17:47:35 2025

@author: user
"""

import streamlit as st
import pandas as pd
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from io import BytesIO

st.set_page_config(page_title="Corner Kick Analysis", layout="wide")

def classify_corner_sequences(df):
    df[['pass_end_x', 'pass_end_y']] = df['pass.end_location'].str.split(',', expand=True).astype(float)

    if 'index' in df.columns:
        df = df.sort_values(by='index').reset_index(drop=True)
    elif 'event_id' in df.columns:
        df = df.sort_values(by='event_id').reset_index(drop=True)
    else:
        st.error("No time-order column found. Please ensure 'index' or 'event_id' is in your dataset.")
        return None, None

    if not pd.api.types.is_timedelta64_dtype(df['timestamp']):
        try:
            df['timestamp'] = pd.to_timedelta(df['timestamp'])
        except Exception as e:
            st.error("Timestamp column could not be converted to timedelta.")
            return None, None

    corner_passes = df[
        (df['type.name'] == 'Pass') &
        (df['play_pattern.name'] == 'From Corner') &
        (df['pass.type.name'] == 'Corner')
    ]

    if corner_passes.empty:
        st.warning("No corner passes found in the dataset.")
        return None, None

    results = []

    for idx, row in corner_passes.iterrows():
        start_index = idx
        start_team = row['possession_team.id']

        if 'location' in row and isinstance(row['location'], (list, tuple)) and len(row['location']) >= 1:
            side = 'Left' if row['location'][0] < 60 else 'Right'
        else:
            side = 'Unknown'

        pass_height = row.get('pass.height.name', 'Unknown')
        pass_body_part = row.get('pass.body_part.name', 'Unknown')
        pass_outcome = row.get('pass.outcome.name', 'Unknown')
        pass_technique = row.get('pass.technique.name', 'Unknown')

        subsequent_events = df.iloc[start_index + 1:]
        same_possession = subsequent_events[subsequent_events['possession_team.id'] == start_team]

        if same_possession.empty:
            classification = 'No first contact - no shot'
            results.append({
                'corner_index': idx, 'classification': classification, 'side': side,
                'pass_height': pass_height, 'pass_body_part': pass_body_part,
                'pass_outcome': pass_outcome, 'pass_technique': pass_technique,
                'pass_end_x': row['pass_end_x'], 'pass_end_y': row['pass_end_y'],
                'team': row['possession_team.name']
            })
            continue

        first_contact = same_possession.iloc[0]

        if first_contact['type.name'] == 'Shot':
            classification = 'First contact - direct shot'
        else:
            shot_within_3_sec = same_possession[
                (same_possession['type.name'] == 'Shot') &
                ((same_possession['timestamp'] - first_contact['timestamp']).dt.total_seconds() <= 3)
            ]
            if not shot_within_3_sec.empty:
                classification = 'First contact - shot within 3 seconds'
            else:
                any_shot = same_possession[same_possession['type.name'] == 'Shot']
                if not any_shot.empty:
                    classification = 'No first contact - shot'
                else:
                    classification = 'First contact - no shot'

        results.append({
            'corner_index': idx, 'classification': classification, 'side': side,
            'pass_height': pass_height, 'pass_body_part': pass_body_part,
            'pass_outcome': pass_outcome, 'pass_technique': pass_technique,
            'pass_end_x': row['pass_end_x'], 'pass_end_y': row['pass_end_y'],
            'team': row['possession_team.name']
        })

    summary_df = pd.DataFrame(results)
    return df, summary_df

def calculate_xg_stats(df, summary_df):
    xg_total = 0.0
    xg_inswinger = 0.0
    xg_outswinger = 0.0
    xg_per_corner = []

    for idx, row in summary_df.iterrows():
        corner_index = row['corner_index']
        possession_team = df.loc[corner_index, 'possession_team.id']
        subsequent_events = df.iloc[corner_index + 1:]
        same_possession = subsequent_events[subsequent_events['possession_team.id'] == possession_team]
        shots = same_possession[same_possession['type.name'] == 'Shot']
        corner_xg = shots['shot.statsbomb_xg'].sum()
        xg_per_corner.append(corner_xg)
        xg_total += corner_xg

        pass_technique = row['pass_technique']
        if isinstance(pass_technique, str):
            if pass_technique.lower() == 'inswinger':
                xg_inswinger += corner_xg
            elif pass_technique.lower() == 'outswinger':
                xg_outswinger += corner_xg

    summary_df['xg_per_corner'] = xg_per_corner
    return xg_total, xg_inswinger, xg_outswinger

def plot_corner_passes(summary_df, xg_total, xg_inswinger, xg_outswinger):
    pitch = VerticalPitch(half=True, pitch_type='statsbomb', line_color='black')
    fig, ax = pitch.draw(figsize=(12, 10))

    markers = {
        'First contact - direct shot': 'o',
        'First contact - shot within 3 seconds': 's',
        'No first contact - shot': '^',
        'First contact - no shot': 'X',
        'No first contact - no shot': 'P'
    }
    colors = {
        'First contact - direct shot': 'red',
        'First contact - shot within 3 seconds': 'blue',
        'No first contact - shot': 'green',
        'First contact - no shot': 'orange',
        'No first contact - no shot': 'gray'
    }

    for classification, marker in markers.items():
        subset = summary_df[summary_df['classification'] == classification]
        x = subset['pass_end_x']
        y = subset['pass_end_y']
        pitch.scatter(x, y, ax=ax, marker=marker, color=colors[classification], label=classification, s=100, edgecolors='black')

    handles = [Line2D([0], [0], marker=m, color='w', label=cls,
                      markerfacecolor=colors[cls], markeredgecolor='black', markersize=10)
               for cls, m in markers.items()]
    leg = ax.legend(handles=handles, title="Classification", loc='center left', bbox_to_anchor=(1.05, 0.5))
    leg.get_frame().set_edgecolor('black')

    stats_text = (
        f'Total xG: {xg_total:.3f}\n'
        f'xG from inswingers: {xg_inswinger:.3f}\n'
        f'xG from outswingers: {xg_outswinger:.3f}'
    )
    plt.text(1.12, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.title('Corner Pass End Locations by Classification', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig

# Streamlit UI
st.title("Corner Kick Sequence Analyzer")
st.write("This app classifies corner sequences and analyzes xG statistics for selected teams.")

try:
    df = pd.read_excel('ger.xlsx')
    st.success("File 'ger.xlsx' loaded successfully!")

    df, summary_df = classify_corner_sequences(df)

    if summary_df is not None:
        team_list = summary_df['team'].dropna().unique()
        selected_team = st.selectbox("Select a Team", team_list)

        team_summary_df = summary_df[summary_df['team'] == selected_team]

        st.subheader(f"Classification Summary for {selected_team}")
        st.write(team_summary_df['classification'].value_counts())
        st.dataframe(team_summary_df)

        xg_total, xg_inswinger, xg_outswinger = calculate_xg_stats(df, team_summary_df)

        fig = plot_corner_passes(team_summary_df, xg_total, xg_inswinger, xg_outswinger)
        st.pyplot(fig)

        st.subheader(f"xG Statistics for {selected_team}")
        st.write(f"**Total xG from corners:** {xg_total:.3f}")
        st.write(f"**xG from inswingers:** {xg_inswinger:.3f}")
        st.write(f"**xG from outswingers:** {xg_outswinger:.3f}")

        buffer = BytesIO()
        team_summary_df.to_excel(buffer, index=False)
        st.download_button(f"Download {selected_team} Results", buffer.getvalue(), file_name=f"{selected_team}_corner_classification.xlsx")

except Exception as e:
    st.error(f"Error processing file: {e}")
