'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

import pandas as pd

def preprocess_data(pred_universe, arrest_events):
    # Show columns and values for debugging
    print("Columns in arrest_events:")
    print(arrest_events.columns.tolist())
    print("Unique values in charge_degree:")
    print(arrest_events['charge_degree'].unique())

    # Merge on person_id
    df_arrests = pd.merge(pred_universe, arrest_events, how="outer", on="person_id")

    # Target variable: y = rearrested for felony within 1 year
    def was_rearrested(row):
        if pd.isnull(row['arrest_date_univ']):
            return 0
        start = row['arrest_date_univ'] + pd.Timedelta(days=1)
        end = row['arrest_date_univ'] + pd.Timedelta(days=365)
        events = arrest_events[
            (arrest_events['person_id'] == row['person_id']) &
            (arrest_events['arrest_date_event'] >= start) &
            (arrest_events['arrest_date_event'] <= end) &
            (arrest_events['charge_degree'].str.lower() == 'felony')
        ]
        return 1 if not events.empty else 0

    df_arrests['y'] = df_arrests.apply(was_rearrested, axis=1)
    print("What share of arrestees were rearrested for a felony crime in the next year?")
    print(df_arrests['y'].mean())

    # Predictive feature: current_charge_felony
    df_arrests['current_charge_felony'] = (
        df_arrests['charge_degree'].str.lower() == 'felony'
    ).astype(int)
    print("What share of current charges are felonies?")
    print(df_arrests['current_charge_felony'].mean())

    # Predictive feature: num_fel_arrests_last_year
    def count_felonies_last_year(row):
        if pd.isnull(row['arrest_date_univ']):
            return 0
        start = row['arrest_date_univ'] - pd.Timedelta(days=365)
        end = row['arrest_date_univ'] - pd.Timedelta(days=1)
        events = arrest_events[
            (arrest_events['person_id'] == row['person_id']) &
            (arrest_events['arrest_date_event'] >= start) &
            (arrest_events['arrest_date_event'] <= end) &
            (arrest_events['charge_degree'].str.lower() == 'felony')
        ]
        return len(events)

    df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(count_felonies_last_year, axis=1)
    print("What is the average number of felony arrests in the last year?")
    print(df_arrests['num_fel_arrests_last_year'].mean())

    # Final preview
    print(df_arrests.head())
    print("Preprocessing complete.")

    return df_arrests






