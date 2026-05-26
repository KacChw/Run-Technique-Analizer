import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def calculate_angle(row):
    # Punkty stawów
    biodro = np.array([row['Biodro_X'], row['Biodro_Y']])
    kolano = np.array([row['Kolano_X'], row['Kolano_Y']]) # Wierzchołek kąta
    kostka = np.array([row['Kostka_X'], row['Kostka_Y']])
    
    # Tworzenie wektorów od kolana do biodra i do kostki
    wektor_biodro = biodro - kolano
    wektor_kostka = kostka - kolano
    
    # Obliczanie cosinusa kąta
    iloczyn_skalarny = np.dot(wektor_biodro, wektor_kostka)
    dlugosc_biodro = np.linalg.norm(wektor_biodro)
    dlugosc_kostka = np.linalg.norm(wektor_kostka)
    
    # Zabezpieczenie przed dzieleniem przez zero (gdy punkty się nałożą)
    if dlugosc_biodro == 0 or dlugosc_kostka == 0:
        return 0.0
        
    cos_theta = iloczyn_skalarny / (dlugosc_biodro * dlugosc_kostka)
    # Zabezpieczenie przed błędami precyzji Pythona (cos_theta musi być w [-1, 1])
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Przeliczenie radianów na stopnie
    kat_radiany = np.arccos(cos_theta)
    return np.degrees(kat_radiany)

# 1. Wczytanie danych
df = pd.read_csv('dane_biegu.csv')

# 2. Wyciągnięcie wycinka nagrania (np. klatki od 500 do 650) 
# Dostosuj te liczby, jeśli akurat w tym momencie na wideo nic się nie działo
df_subset = df.iloc[600:650].copy()

# 3. Obliczenie kąta dla każdej klatki
df_subset['Kat_Kolana'] = df_subset.apply(calculate_angle, axis=1)

# 4. Wygładzenie wykresu kąta (zwiększone okno do 15 dla lepszego efektu)
df_subset['Kat_Kolana_Wygładzony'] = savgol_filter(df_subset['Kat_Kolana'], window_length=15, polyorder=3)

# 5. Rysowanie wykresu
plt.figure(figsize=(10, 5))
plt.plot(df_subset['Klatka'], df_subset['Kat_Kolana_Wygładzony'], color='green', linewidth=2, label='Kąt kolana')

# Dodanie punktów minimalnych (maksymalne zgięcie nogi) na wykresie
plt.title('Kąt zgięcia kolana w czasie (zbliżenie na pojedyncze kroki)')
plt.xlabel('Numer klatki')
plt.ylabel('Kąt (stopnie)')
plt.grid(True)
plt.legend()
plt.show()