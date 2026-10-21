import os
import cv2
import threading
from datetime import datetime

from flask import Flask, Response, render_template_string
from ultralytics import YOLO

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

os.makedirs("capturas_ingresos", exist_ok=True)

print("Cargando modelo YOLO...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

# Línea virtual
y_linea = 300
x_inicio = 100
x_fin = 540

ids_cruzados = set()
posiciones_previas = {}

contador_ingresos = 0
ultimo_frame = None

lock = threading.Lock()

app = Flask(__name__)


# --------------------------------------------------
# PROCESAMIENTO DE CÁMARA + YOLO
# --------------------------------------------------

def procesar_camara():
    global ultimo_frame
    global contador_ingresos

    print("Sistema de detección iniciado...")

    while True:

        success, frame = cap.read()

        if not success:
            print("No se pudo leer el fotograma.")
            continue

        # Seguimiento de personas
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.3,
            verbose=False
        )

        # Línea virtual
        cv2.line(
            frame,
            (x_inicio, y_linea),
            (x_fin, y_linea),
            (0, 0, 255),
            3
        )

        # Procesar detecciones
        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()

            track_ids = (
                results[0].boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            for box, track_id in zip(boxes, track_ids):

                x1, y1, x2, y2 = map(int, box)

                # Punto inferior central
                cx = int((x1 + x2) / 2)
                cy = y2

                # Caja
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # ------------------------------------
                # DETECCIÓN DEL CRUCE
                # ------------------------------------

                if track_id in posiciones_previas:

                    y_anterior = posiciones_previas[track_id]

                    # Arriba -> abajo
                    if (
                        y_anterior < y_linea
                        and cy >= y_linea
                        and track_id not in ids_cruzados
                    ):

                        if x_inicio <= cx <= x_fin:

                            contador_ingresos += 1
                            ids_cruzados.add(track_id)

                            fecha = datetime.now().strftime(
                                "%Y%m%d_%H%M%S"
                            )

                            nombre_archivo = (
                                f"capturas_ingresos/"
                                f"ingreso_{fecha}_id_{track_id}.jpg"
                            )

                            cv2.imwrite(
                                nombre_archivo,
                                frame
                            )

                            print(
                                "[ACCESO REGISTRADO]",
                                nombre_archivo
                            )

                posiciones_previas[track_id] = cy

        # ------------------------------------
        # CONTADOR
        # ------------------------------------

        cv2.putText(
            frame,
            f"Ingresos Totales: {contador_ingresos}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        # Convertir frame a JPG para enviarlo por web
        ret, buffer = cv2.imencode(".jpg", frame)

        if ret:
            with lock:
                ultimo_frame = buffer.tobytes()


# --------------------------------------------------
# STREAM DE VÍDEO
# --------------------------------------------------

def generar_video():

    global ultimo_frame

    while True:

        with lock:
            frame = ultimo_frame

        if frame is None:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


# --------------------------------------------------
# PÁGINA WEB
# --------------------------------------------------

pagina = """
<!DOCTYPE html>

<html>

<head>

<title>Control de Ingreso - Herbario BAB</title>

<style>

body {
    background:#111;
    color:white;
    font-family:Arial;
    text-align:center;
}

h1 {
    color:#7ddc7d;
}

img {
    width:90%;
    max-width:1000px;
    border:3px solid #555;
    border-radius:10px;
}

.contador {
    font-size:30px;
    margin:20px;
}

</style>

</head>

<body>

<h1>Control de Ingreso - Herbario BAB</h1>

<div class="contador">
Ingresos detectados: {{ contador }}
</div>

<img src="/video">

</body>

</html>
"""


@app.route("/")
def index():

    return render_template_string(
        pagina,
        contador=contador_ingresos
    )


@app.route("/video")
def video():

    return Response(
        generar_video(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


# --------------------------------------------------
# INICIO
# --------------------------------------------------

if __name__ == "__main__":

    hilo = threading.Thread(
        target=procesar_camara,
        daemon=True
    )

    hilo.start()

    print("")
    print("Servidor iniciado.")
    print("Desde otra PC abre:")
    print("http://IP_DE_LA_NOTEBOOK:5000")
    print("")

    app.run(
        host="localhost",
        port=5000,
        debug=False,
        threaded=True
    )