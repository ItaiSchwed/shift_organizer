from datetime import timedelta, datetime

import numpy as np
import streamlit as st
from scheduler import get_schedule
import pandas as pd

st.set_page_config(layout="wide")

day_number_to_name = ['שני', 'שלישי', 'רביעי', 'חמישי', 'שישי', 'שבת', 'ראשון']


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
    df.rating = df.rating.astype(object)

    return df, dates


left_shift, centered_screen, right_shift = st.columns([1, 3, 1])

with centered_screen:
    # Streamlit UI
    st.title("חלוקת תורנויות")

    # File uploader widget
    uploaded_file = st.file_uploader("בחר קובץ", type=["xlsx"])


def highlight_cells(sched, highlight_dict, color, existing_styles=None):
    # Create a DataFrame for styles if not provided
    if existing_styles is None:
        style_df = pd.DataFrame('', index=sched.index, columns=sched.columns)
    else:
        style_df = existing_styles.copy()  # Copy existing styles to avoid overwriting

    for index, dates in highlight_dict.items():
        for date in dates:
            # Set background color with alpha (0.5)
            style_df.at[index, get_date_multicol(date)] = f'background-color: rgba({color}, 0.3)'
    return style_df


def get_styled_schedule(schedule):
    # Replace None with empty strings for display
    schedule = schedule.fillna("")
    # Convert numbers to integers where possible and leave empty cells as they are
    schedule = schedule.map(lambda x: int(x) if isinstance(x, (float, int)) and not pd.isna(x) else "")

    red = '255, 0, 0'
    green = '0, 255, 0'
    yellow = '255, 180, 0'

    # Modify the way you call the highlight_cells function to use RGB values
    styles = highlight_cells(schedule, metadata['dates_off'], red)
    styles = highlight_cells(schedule, metadata['prefer_working'], green, styles)
    styles = highlight_cells(schedule, metadata['prefer_not_working'], yellow, styles)
    # Apply the styling for the schedule DataFrame
    return schedule.style.apply(lambda _: styles, axis=None)


def show_statistics_and_scores():
    # Apply color styling for the statistics DataFrame
    styled_statistics = statistics.style.background_gradient(cmap="YlGnBu", axis=None)
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
            st.dataframe(styled_statistics, use_container_width=True)
    with col2:
        with st.columns(3)[1]:
            st.subheader('ציונים')
            scores_to_show = pd.Series(scores, name='ציון').to_frame()
            scores_to_show = (scores_to_show * 100).round(2).astype(str) + '%'
            st.dataframe(scores_to_show, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def get_date_multicol(date):
    return date.strftime("%d-%m"), day_number_to_name[date.weekday()]


if uploaded_file:
    # Load the Excel file into a DataFrame
    df = pd.read_excel(uploaded_file)

    with centered_screen:
        st.write("הקובץ נטען בהצלחה")

        with st.expander('קונפיגורציה'):
            freedom = st.number_input('הפרש ימים מקסימלי בין עובדים', min_value=0, value=1, step=1)
            st.divider()
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
            with col1:
                sunday_to_wednesday_weight = st.number_input('משקל לראשון עד רביעי', min_value=0, value=1, step=1)
            with col2:
                thursday_weight = st.number_input('משקל לחמישי', min_value=0, value=2, step=1)
            with col3:
                friday_saturday_weight = st.number_input('משקל לשישי ושבת', min_value=0, value=3, step=1)
            with col4:
                st.divider()
            with col5:
                weighted_freedom = st.number_input('הפרש ממושקל מקסימלי בין עובדים', min_value=0, value=1, step=1)

            weekday_weights = {
                6: sunday_to_wednesday_weight,
                0: sunday_to_wednesday_weight,
                1: sunday_to_wednesday_weight,
                2: sunday_to_wednesday_weight,
                3: thursday_weight,
                4: friday_saturday_weight,
                5: friday_saturday_weight
            }

            st.divider()
            max_shift_distance = st.number_input('הפער המקסימלי בין דרגה לתורנות', min_value=0, value=2, step=1)

        # Process the DataFrame
        df, dates = process_excel(df)

        calculate = st.button('חשב')

    if calculate:
        with centered_screen:
            with st.spinner('calculating...'):
                schedule, statistics, metadata, scores = get_schedule(df, dates,
                                                                      max_shift_distance=max_shift_distance,
                                                                      freedom=freedom,
                                                                      weighted_freedom=weighted_freedom,
                                                                      weekday_weights=weekday_weights)
        if schedule is not None:

            # Create the multi-index by combining date and day of the week
            multi_index = pd.MultiIndex.from_tuples([get_date_multicol(date)
                                                     for date in schedule.columns], names=["Date", "Day of Week"])

            # Assign the multi-index to the columns
            schedule.columns = multi_index

            st.dataframe(get_styled_schedule(schedule))
            show_statistics_and_scores()
        else:
            with centered_screen:
                st.write("לא נמצאה הצבה העונה על כלל הדרישות")

else:
    with centered_screen:
        st.write("אנא העלה קובץ אקסל על מנת להמשיך.")
