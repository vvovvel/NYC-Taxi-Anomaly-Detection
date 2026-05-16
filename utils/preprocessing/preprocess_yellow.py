import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_yellow(df, lookup_path="../data/taxi_zone_lookup.csv"):
    """
    Wykonuje wstępne przetwarzanie surowych danych dla żółtych taksówek.
    Obejmuje selekcję kolumn, czyszczenie anomalii, inżynierię cech i kodowanie geograficzne.
    """
    # Selekcja kolumn
    columns_to_keep = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count',
        'trip_distance', 'RatecodeID', 'payment_type', 'PULocationID', 'DOLocationID',
        'fare_amount', 'tip_amount', 'tolls_amount',
        'total_amount', 'congestion_surcharge', 'Airport_fee'
    ]
    df = df[columns_to_keep].copy()

    # Oczyszczenie z braków danych (NaN) przed konwersją typów
    columns_with_nan = ['passenger_count', 'RatecodeID', 'payment_type', 'congestion_surcharge', 'Airport_fee']
    df = df.dropna(subset=columns_with_nan)

    # Konwersja typów numerycznych dla optymalizacji pamięci
    target_dtypes = {
        'passenger_count': 'int8',
        'trip_distance': 'float32',
        'RatecodeID': 'int8',
        'payment_type': 'int8',
        'PULocationID': 'int16',
        'DOLocationID': 'int16',
        'fare_amount': 'float32',
        'tip_amount': 'float32',
        'tolls_amount': 'float32',
        'total_amount': 'float32',
        'congestion_surcharge': 'float32',
        'Airport_fee': 'float32'
    }
    df = df.astype(target_dtypes)

    # Konwersja znaczników czasu
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    # Inżynieria cech
    df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0
    df['duration'] = df['duration'].astype('float32')
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour.astype('int8')
    df['pickup_day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek.astype('int8')

    # Filtrowanie błędów logicznych i anomalii
    correct_duration = (df['duration'] >= 1.0) & (df['duration'] <= 360.0)
    correct_distance = df['trip_distance'] > 0.0
    correct_fare = df['fare_amount'] > 0.0
    correct_total = df['total_amount'] > 0.0
    df = df[correct_duration & correct_distance & correct_fare & correct_total]

    # Mapowanie stref na dzielnice (Boroughs) i One-Hot Encoding
    lookup_df = pd.read_csv(lookup_path)
    lookup_df['LocationID'] = lookup_df['LocationID'].astype('int16')
    borough_map = lookup_df[['LocationID', 'Borough']]

    df = df.merge(borough_map, left_on='PULocationID', right_on='LocationID', how='left')
    df = df.rename(columns={'Borough': 'PU_Borough'}).drop(columns=['LocationID'])

    df = df.merge(borough_map, left_on='DOLocationID', right_on='LocationID', how='left')
    df = df.rename(columns={'Borough': 'DO_Borough'}).drop(columns=['LocationID'])

    categorical_cols = ['PU_Borough', 'DO_Borough']
    df_dummies = pd.get_dummies(df[categorical_cols], prefix=['PU', 'DO'], dtype='int8')
    df = pd.concat([df, df_dummies], axis=1)

    # Usunięcie identyfikatorów stref, ale zachowanie nazw tekstowych (PU_Borough, DO_Borough)
    df = df.drop(columns=['PULocationID', 'DOLocationID'])

    return df


def sample_data(df, n_samples=25000, random_state=42):
    """
    Pobiera losową, reprezentatywną próbkę danych z zachowaniem oryginalnego indeksu.
    """
    if len(df) <= n_samples:
        return df.copy()
    return df.sample(n=n_samples, random_state=random_state).copy()


def scale_data(df_sample):
    """
    Wyodrębnia cechy numeryczne i poddaje je standaryzacji (StandardScaler).
    """
    # MUSIMY wykluczyć PU_Borough i DO_Borough, bo to tekst (stringi)
    exclude_cols = [
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime',
        'pickup_hour',
        'pickup_day_of_week',
        'total_amount',
        'RatecodeID',
        'payment_type',
        'PU_Borough',
        'DO_Borough'
    ]

    # Tylko kolumny, których nie ma na liście wykluczeń, trafią do skalowania
    feature_cols = [col for col in df_sample.columns if col not in exclude_cols]

    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(df_sample[feature_cols])

    # Tworzymy DataFrame z zachowaniem nazw kolumn i indeksów
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=feature_cols, index=df_sample.index)

    return X_scaled_df, scaler