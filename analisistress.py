import cv2
from deepface import DeepFace
import numpy as np

# --- Variabel untuk Optimasi Kinerja ---
# Hanya proses setiap N frame untuk mengurangi beban kerja
PROCESS_EVERY_N_FRAMES = 15 
# Ubah angka ini sesuai kebutuhan:
# Angka lebih tinggi = lebih ringan tapi update lebih jarang
# Angka lebih rendah = lebih responsif tapi lebih berat

frame_counter = 0

# Variabel untuk menyimpan hasil terakhir, agar tidak berkedip
last_gender = "N/A"
last_emotion = "N/A"
last_stress_score = 0
last_advice = ""
# ----------------------------------------

# Tips stres
stress_tips = [
    "Coba tarik napas dalam-dalam.",
    "Istirahat sejenak dari layar.",
    "Dengarkan musik yang menenangkan.",
    "Coba meditasi atau berjalan kaki.",
    "Hubungi orang terdekat untuk cerita."
]

# Fungsi untuk menghitung skor stres berdasarkan emosi
def get_stress_score(emotion):
    stress_emotions = ['angry', 'sad', 'fear', 'disgust']
    if emotion in stress_emotions:
        return 75 + stress_emotions.index(emotion) * 5
    elif emotion == 'neutral':
        return 30
    else:
        return 10

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    
    # --- Blok Peningkatan Kinerja ---
    # Hanya jalankan analisis jika sudah waktunya (setiap N frame)
    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        try:
            # 1. Kecilkan frame untuk analisis yang lebih cepat
            # Kita analisis frame kecil, tapi tampilkan di frame besar asli
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # 2. Jalankan analisis pada frame yang sudah dikecilkan
            analysis = DeepFace.analyze(small_frame, actions=['gender', 'emotion'], enforce_detection=False)

            # Update hasil terakhir
            gender_en = analysis[0]['dominant_gender']
            last_gender = "Pria" if gender_en == "Man" else "Wanita"
            last_emotion = analysis[0]['dominant_emotion']
            last_stress_score = get_stress_score(last_emotion)

            if last_stress_score >= 50:
                last_advice = np.random.choice(stress_tips)
            else:
                last_advice = ""

        except Exception as e:
            # Jika tidak ada wajah terdeteksi, tidak perlu update apa-apa
            pass
    # ---------------------------------

    # Siapkan teks untuk ditampilkan (menggunakan data terakhir)
    result_text = f"Gender: {last_gender} | Emosi: {last_emotion} | Stres: {last_stress_score}%"

    # Selalu tampilkan hasil terakhir di setiap frame agar tidak berkedip
    cv2.putText(frame, result_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    if last_advice:
        cv2.putText(frame, f"Saran: {last_advice}", (20, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

    # Tampilkan jendela video
    cv2.imshow("Stress Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()