###############################################################
# CONFIGURAZIONE MODALITÀ
###############################################################
DEBUG_MODE = True   # True allora debug con finestra; False allora gara senza GUI
CAM = True          # True allora camera attacata altrimenti camera computer base


import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

model = load_model("greek_letters_model_folder")

###############################################################
# CLASSI DEL MODELLO (ordine identico al training)
###############################################################
GREEK_CLASSES = ["PHI", "PSI", "OMEGA"]


# Soglia minima di confidenza
CONF_THRESHOLD = 0.80

# Stabilizzazione: numero di frame da considerare
STABILITY_FRAMES = 5
history = deque(maxlen=STABILITY_FRAMES)


###############################################################
# PREPROCESSING (identico al training)
###############################################################
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    norm = resized.astype("float32") / 255.0
    norm = np.expand_dims(norm, axis=-1)
    norm = np.expand_dims(norm, axis=0)
    return norm


###############################################################
# RICONOSCIMENTO SINGOLO FRAME
###############################################################
def recognize_single(frame):
    inp = preprocess(frame)
    pred = model.predict(inp, verbose=0)[0]

    idx = np.argmax(pred)
    prob = pred[idx]

    if prob >= CONF_THRESHOLD:
        return GREEK_CLASSES[idx], prob
    return "NULL", prob


###############################################################
# FILTRO DI STABILIZZAZIONE
# Restituisce una lettera solo se appare coerente su più frame
###############################################################
def stabilized_recognition(letter):
    history.append(letter)

    if history.count(letter) >= STABILITY_FRAMES * 0.6 and letter != "NULL":
        return letter

    return "NULL"


###############################################################
# LOOP PRINCIPALE
###############################################################

if CAM:
    tmp = 0
else:
    tmp = 1

cap = cv2.VideoCapture(tmp, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Errore: impossibile aprire la webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Errore: frame non valido.")
        break

    if DEBUG_MODE:
        cv2.imshow("DEBUG - Webcam", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            letter, prob = recognize_single(frame)
            stable = stabilized_recognition(letter)

            if stable != "NULL":
                print(f"[DEBUG] Lettera stabile: {stable} (conf={prob:.2f})")
                cv2.putText(frame, f"{stable} ({prob:.2f})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            else:
                print(f"[DEBUG] Nessuna lettera (conf={prob:.2f})")
                cv2.putText(frame, "NULL", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)

            cv2.imshow("DEBUG - Webcam", frame)

        if key == ord('q'):
            break

    else:
        # Modalità gara: nessuna finestra, riconoscimento continuo
        letter, prob = recognize_single(frame)
        stable = stabilized_recognition(letter)

        if stable != "NULL":
            print(stable)
            break


cap.release()
if DEBUG_MODE:
    cv2.destroyAllWindows()
