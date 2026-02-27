###############################################################
# CONFIGURAZIONE
###############################################################
DEBUG_MODE = True  # True = debug con finestra; False = gara
USE_TFLITE = True  # True = usa modello TFLite, False = usa Keras
CAM = True  # True = camera esterna, False = webcam PC

import cv2
import numpy as np
from collections import deque
import time

###############################################################
# CLASSI DEL MODELLO (ordine identico al training)
###############################################################
GREEK_CLASSES = ["PHI", "PSI", "OMEGA", "EMPTY"]

CONF_THRESHOLD = 0.80
STABILITY_FRAMES = 5
history = deque(maxlen=STABILITY_FRAMES)

###############################################################
# CARICAMENTO MODELLO
###############################################################
if USE_TFLITE:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path="greek_letters_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    from tensorflow.keras.models import load_model

    model = load_model("best_model.keras")


###############################################################
# PREPROCESSING
###############################################################
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    norm = resized.astype("float32") / 255.0
    norm = np.expand_dims(norm, axis=-1)
    norm = np.expand_dims(norm, axis=0)
    return norm


###############################################################
# PREDIZIONE SINGOLO FRAME
###############################################################
def predict_frame(frame):
    inp = preprocess(frame)

    if USE_TFLITE:
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0]
    else:
        pred = model.predict(inp, verbose=0)[0]

    idx = np.argmax(pred)
    prob = pred[idx]

    if prob >= CONF_THRESHOLD:
        return GREEK_CLASSES[idx], prob
    return "NULL", prob


###############################################################
# STABILIZZAZIONE
###############################################################
def stabilized(letter):
    history.append(letter)

    # deve apparire almeno 5 volte su 5
    if history.count(letter) >= STABILITY_FRAMES and letter != "NULL":
        return letter

    return "NULL"


###############################################################
# LOOP PRINCIPALE
###############################################################
if CAM:
    cam_index = 0
else:
    cam_index = 1

cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Errore: impossibile aprire la webcam.")
    exit()

print("Sistema pronto. Premi C per riconoscere (debug) oppure avvia gara.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Errore: frame non valido.")
        break

    if DEBUG_MODE:
        cv2.imshow("DEBUG - Webcam", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            letter, prob = predict_frame(frame)
            stable = stabilized(letter)

            if stable != "NULL":
                print(f"[STABILE] {stable} (conf={prob:.2f})")
                cv2.putText(frame, f"{stable} ({prob:.2f})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            else:
                print(f"[NO] {letter} (conf={prob:.2f})")
                cv2.putText(frame, "NULL", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            cv2.imshow("DEBUG - Webcam", frame)

        if key == ord('q'):
            break

    else:
        # Modalità gara: nessuna finestra, riconoscimento continuo
        letter, prob = predict_frame(frame)
        stable = stabilized(letter)

        if stable != "NULL":
            print(stable)
            break

cap.release()
if DEBUG_MODE:
    cv2.destroyAllWindows()