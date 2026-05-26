import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 1. Wczytanie danych z pliku CSV
df = pd.read_csv('dane_biegu.csv')

# 2. Zastosowanie filtra Savitzky'ego-Golaya dla pozycji Y kolana
# window_length: rozmiar okna (musi być liczbą nieparzystą, np. 11 klatek)
# polyorder: stopień wielomianu (zazwyczaj 2 lub 3)
window = 11
poly = 3

df['Kolano_Y_Wygładzone'] = savgol_filter(df['Kolano_Y'], window_length=window, polyorder=poly)

# 3. Rysowanie wykresu porównawczego
plt.figure(figsize=(12, 6))

# Surowe dane (niebieska, przerywana linia)
plt.plot(df['Klatka'], df['Kolano_Y'], label='Surowe dane (YOLO)', color='blue', alpha=0.4, linestyle='--')

# Wygładzone dane (czerwona, ciągła linia)
plt.plot(df['Klatka'], df['Kolano_Y_Wygładzone'], label='Dane wygładzone (Savgol)', color='red', linewidth=2)

plt.title('Pozycja pionowa kolana w czasie (Oś Y)')
plt.xlabel('Numer klatki wideo')
plt.ylabel('Pozycja Y (piksele)')
plt.gca().invert_yaxis() # Odwracamy oś Y, bo w grafice komputerowej piksel 0 jest na samej górze ekranu
plt.legend()
plt.grid(True)

# Wyświetlenie wykresu
plt.show()