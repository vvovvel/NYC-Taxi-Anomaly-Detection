import pandas as pd
import numpy as np
import altair as alt

def plot_altair(df_plot, x_col, y_col, color_col='trip_distance', title="Wizualizacja"):
    """
    Tworzy interaktywny wykres dla dowolnej metody redukcji wymiarów.
    """
    alt.data_transformers.disable_max_rows()

    # Dynamiczne tworzenie etykiet na podstawie nazw kolumn
    chart = alt.Chart(df_plot).mark_circle(size=20, opacity=0.5).encode(
        x=alt.X(f'{x_col}:Q', title=x_col),
        y=alt.Y(f'{y_col}:Q', title=y_col),
        color=alt.Color(f'{color_col}:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=[
            x_col,
            y_col,
            'trip_distance',
            'fare_amount',
            'pickup_hour',
            'PU_Borough',
            'DO_Borough'
        ]
    ).properties(
        width=700,
        height=500,
        title=title
    ).interactive()

    return chart