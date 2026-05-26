import cv2
import csv
from ultralytics import YOLO

# Inicjalizacja modelu i wideo
model = YOLO('yolov8n-pose.pt') 
video_path = "bok.mp4" # Pamiętaj o swojej nazwie pliku
cap = cv2.VideoCapture(video_path)

# 1. Przygotowanie pliku CSV do zapisu danych
csv_filename = 'dane_biegu.csv'
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)

# Zapisujemy nagłówki kolumn w pliku
csv_writer.writerow(['Klatka', 'Biodro_X', 'Biodro_Y', 'Kolano_X', 'Kolano_Y', 'Kostka_X', 'Kostka_Y'])

frame_idx = 0 # Licznik klatek

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        results = model(frame)
        
        # 2. Ekstrakcja punktów (keypoints)
        # Bierzemy dane dla pierwszej wykrytej osoby [0] i wyciągamy współrzędne XY
        keypoints = results[0].keypoints.xy[0] 
        
        # Sprawdzamy, czy model wykrył jakiekolwiek punkty w danej klatce
        if len(keypoints) > 0:
            # Pobranie współrzędnych konkretnych stawów (format: [X, Y])
            biodro = keypoints[12]
            kolano = keypoints[14]
            kostka = keypoints[16]
            
            # 3. Zapis do pliku CSV (rzutujemy na typ float)
            csv_writer.writerow([
                frame_idx,
                float(biodro[0]), float(biodro[1]),
                float(kolano[0]), float(kolano[1]),
                float(kostka[0]), float(kostka[1])
            ])
        
        # Rysowanie i wyświetlanie (jak poprzednio)
        annotated_frame = results[0].plot()
        
        scale_percent = 40 
        width = int(annotated_frame.shape[1] * scale_percent / 100)
        height = int(annotated_frame.shape[0] * scale_percent / 100)
        annotated_frame = cv2.resize(annotated_frame, (width, height))
        
        cv2.imshow("Analiza techniki biegu - YOLO Pose", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
        frame_idx += 1 # Zwiększenie licznika klatek
    else:
        break

# Zamknięcie wideo, okien i pliku CSV
cap.release()
cv2.destroyAllWindows()
csv_file.close()

print(f"Gotowe! Dane zostały zapisane do pliku {csv_filename}")