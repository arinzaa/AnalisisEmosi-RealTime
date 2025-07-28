import cv2
from deepface import DeepFace
import numpy as np
import tkinter as tk
from tkinter import filedialog

stress_tips = [
    "Coba tarik napas dalam-dalam.",
    "Istirahat sejenak dari pekerjaan.",
    "Dengarkan musik yang menenangkan.",
    "Coba meditasi atau berjalan kaki.",
    "Hubungi orang terdekat untuk cerita."
]

def get_stress_score(emotion):
    stress_emotions = ['angry', 'sad', 'fear', 'disgust']
    if emotion in stress_emotions:
        return 75 + stress_emotions.index(emotion) * 5
    elif emotion == 'neutral':
        return 30
    else:
        return 10

def analyze_image_from_gallery():
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(
        title="Pilih Gambar Wajah",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not image_path:
        print("Tidak ada file yang dipilih. Program berhenti.")
        return

    # Muat gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Gagal memuat gambar dari path: {image_path}")
        return

    # === BAGIAN PENTING UNTUK MENGHINDARI KESAN FREEZE ===
    # 1. Segera tampilkan gambar dengan pesan "Loading"
    loading_image = image.copy() # Buat salinan agar gambar asli tidak berubah
    loading_text = "Menganalisis, mohon tunggu..."
    cv2.putText(loading_image, loading_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2) # Warna oranye
    cv2.imshow("Hasil Analisis", loading_image)
    cv2.waitKey(1) # Perintah penting untuk memaksa jendela diperbarui SEKARANG
    # =======================================================

    result_text = "Analisis gagal."
    advice = ""

    try:
        # 2. Lakukan analisis yang berat. Jendela akan menjeda di sini.
        print("Memulai analisis DeepFace...")
        analysis = DeepFace.analyze(image_path, actions=['gender', 'emotion'], enforce_detection=True)

        # Ambil hasil analisis
        gender_en = analysis[0]['dominant_gender']
        emotion = analysis[0]['dominant_emotion']
        
        gender = "Pria" if gender_en == "Man" else "Wanita"
        stress_score = get_stress_score(emotion)

        result_text = f"Gender: {gender} | Emosi: {emotion} | Stres: {stress_score}%"
        print("Analisis berhasil.")

        if stress_score >= 50:
            advice = np.random.choice(stress_tips)

    except ValueError as e:
        result_text = "Wajah tidak dapat terdeteksi di gambar ini."
        print(f"Error analisis: {e}")
    except Exception as e:
        result_text = "Terjadi error saat analisis."
        print(f"Error tak terduga: {e}")

    # 3. Setelah analisis selesai, tampilkan hasil akhir pada gambar asli
    final_image = image.copy()
    cv2.putText(final_image, result_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if advice:
        cv2.putText(final_image, f"Saran: {advice}", (20, final_image.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

    # Perbarui jendela dengan gambar hasil akhir
    cv2.imshow("Hasil Analisis", final_image)
    print("Analisis selesai. Tekan tombol apapun untuk keluar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Jalankan Fungsi Utama ---
if __name__ == "__main__":
    analyze_image_from_gallery()