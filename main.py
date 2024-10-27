from datetime import timedelta, datetime

import numpy as np
import streamlit as st
from scheduler import get_schedule
import pandas as pd

st.set_page_config(layout="wide")


def get_date_list(start_date, end_date):
    """Returns a list of all dates between the start and end dates (inclusive)."""
    delta = end_date - start_date
    return [start_date + timedelta(days=i) for i in range(delta.days + 1)]


# Define the method to process the Excel file
def process_excel(df):
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("בחר תאריך התחלה", value=datetime(2024, 10, 1))
    with col2:
        end_date = st.date_input("בחר תאריך סיום", value=datetime(2024, 10, 30))
    dates = get_date_list(start_date, end_date)

    df.columns = ['name', 'rating', *dates, 'notes']

    return df, dates


col1, col2, col3 = st.columns(3)

with col2:
    # Streamlit UI
    st.title("חלוקת תורנויות")

    # File uploader widget
    uploaded_file = st.file_uploader("בחר קובץ", type=["xlsx"])

if uploaded_file:
    # Load the Excel file into a DataFrame
    df = pd.read_excel(uploaded_file)

    with col2:
        st.write("הקובץ נטען בהצלחה")

        # Process the DataFrame
        df, dates = process_excel(df)

        calculate = st.button('חשב')

    if calculate:
        schedule, statistics, metadata, scores = get_schedule(df, dates)
        if schedule is not None:
            # Replace None with empty strings for display
            schedule = schedule.fillna("")

            # Convert numbers to integers where possible and leave empty cells as they are
            schedule = schedule.applymap(lambda x: int(x) if isinstance(x, (float, int)) and not pd.isna(x) else "")


            def highlight_cells(sched, highlight_dict, color, existing_styles=None):
                # Create a DataFrame for styles if not provided
                if existing_styles is None:
                    style_df = pd.DataFrame('', index=sched.index, columns=sched.columns)
                else:
                    style_df = existing_styles.copy()  # Copy existing styles to avoid overwriting

                for row_key, cols in highlight_dict.items():
                    if row_key in sched.index:
                        for col in cols:
                            if col in sched.columns:
                                # Set background color with alpha (0.5)
                                style_df.at[row_key, col] = f'background-color: rgba({color}, 0.3)'
                return style_df


            # Modify the way you call the highlight_cells function to use RGB values
            styles = highlight_cells(schedule, metadata['dates_off'], '255, 0, 0')  # Red
            styles = highlight_cells(schedule, metadata['prefer_working'], '0, 255, 0', styles)  # Green
            styles = highlight_cells(schedule, metadata['prefer_not_working'], '255, 180, 0', styles)  # Yellow

            # Apply the styling for the schedule DataFrame
            schedule_style = schedule.style.apply(lambda _: styles, axis=None)
            st.dataframe(schedule_style)

            # Apply color styling for the statistics DataFrame
            statistics_style = statistics.style.background_gradient(cmap="YlGnBu", axis=None)

            st.markdown(
                """
                <style>
                .rtl .stDataFrame {
                    direction: RTL;
                    text-align: right;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Wrap dataframes in an HTML div to apply the RTL class
            st.markdown('<div class="rtl">', unsafe_allow_html=True)
            # Display dataframes in the middle column with RTL styling applied
            col1, col2 = st.columns(2)
            with col1:
                with st.columns([1, 2])[1]:
                    st.subheader('סך הכל')
                    st.dataframe(statistics_style, use_container_width=True)
            with col2:
                with st.columns(3)[1]:
                    st.subheader('ציונים')
                    st.dataframe(pd.Series(scores, name='ציון').to_frame(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            with col2:
                st.write("לא נמצאה הצבה העונה על כלל הדרישות")

else:
    with col2:
        st.write("אנא העלה קובץ אקסל על מנת להמשיך.")
