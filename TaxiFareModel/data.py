import pandas as pd
from TaxiFareModel.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH

def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", nrows=nrows)
    return df


def clean_data(df, test=False):
    ''' 
        -+-+ Clean the data +-+-
        - Drop NaN values
        - Ignore entries with the dropoff/pickup longitude/latitude = 0, why?
        - 0 < Fare Amount < 100
        - 0 <= Passengers < 8
        - Filter area of pickup/dropoff
    '''
    df = df.dropna(how='any', axis='rows')
    # What is the point of these lines whenwe filter more specifically below ?
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


if __name__ == '__main__':
    # Return the data frame from the data stored on AWS
    df = get_data()
