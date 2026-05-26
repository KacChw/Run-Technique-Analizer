import cv2
from ultralytics import YOLO

# 1. Wczytanie modelu AI
# Używamy wersji 'n' (nano) - yolov8n-pose.pt. Jest najszybsza do testów.
# Przy pierwszym uruchomieniu plik modelu sam się pobierze z internetu.
model = YOLO('yolov8n-pose.pt') 

# 2. Wczytanie pliku wideo
video_path = "bok.mp4" # <-- TUTAJ WPISZ NAZWĘ SWOJEGO PLIKU
cap = cv2.VideoCapture(video_path)

# Pętla odtwarzająca wideo klatka po klatce
while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        # 3. Przepuszczenie klatki przez model AI
        # model zwróci obiekt 'results' ze współrzędnymi stawów
        results = model(frame)
        
        # 4. Magia Ultralytics: gotowa funkcja plot() sama rysuje szkielet na obrazie
        annotated_frame = results[0].plot()
        
        # Opcjonalne: Zmiana rozmiaru okna, żeby zmieściło się na monitorze
        # Ustawiamy skalę, np. na 40% oryginalnej wielkości 
# (możesz zmienić tę wartość na 30 lub 50, żeby dopasować do swojego monitora)
        scale_percent = 40 

        width = int(annotated_frame.shape[1] * scale_percent / 100)
        height = int(annotated_frame.shape[0] * scale_percent / 100)
        dim = (width, height)

# Zmiana rozmiaru z zachowaniem oryginalnych proporcji
        annotated_frame = cv2.resize(annotated_frame, dim)
        
        # 5. Wyświetlenie gotowej klatki na ekranie
        cv2.imshow("Analiza techniki biegu - YOLO Pose", annotated_frame)
        
        # Przerwij wciskając klawisz 'q' na klawiaturze
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Pętla kończy się, gdy skończy się wideo
        break

# 6. Sprzątanie i zamknięcie okien
cap.release()
cv2.destroyAllWindows()