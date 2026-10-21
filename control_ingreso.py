import os
import cv2
import threading
from datetime import datetime

from flask import Flask, Response, render_template_string, jsonify
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================

CARPETA_CAPTURAS = "capturas_accesos"
os.makedirs(CARPETA_CAPTURAS, exist_ok=True)

# Línea virtual
Y_LINEA = 300
X_INICIO = 100
X_FIN = 540

# Margen alrededor de la línea para evitar oscilaciones
MARGEN = 10

# Puerto del servidor web
PUERTO = 5000


# ============================================================
# YOLO
# ============================================================

print("Cargando modelo YOLO...")

model = YOLO("yolov8n.pt")


# ============================================================
# CÁMARA
# ============================================================

print("Abriendo cámara...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("ERROR: No se pudo abrir la cámara.")


# ============================================================
# VARIABLES
# ============================================================

posiciones_previas = {}

contador_ingresos = 0
contador_egresos = 0

ultimo_frame = None

ultimos_eventos = []

lock = threading.Lock()

app = Flask(__name__)


# ============================================================
# REGISTRAR EVENTO
# ============================================================

def registrar_evento(tipo, track_id, frame):

    ahora = datetime.now()

    fecha_archivo = ahora.strftime("%Y%m%d_%H%M%S")

    nombre_archivo = (
        f"{CARPETA_CAPTURAS}/"
        f"{tipo.lower()}_{fecha_archivo}_id_{track_id}.jpg"
    )

    cv2.imwrite(nombre_archivo, frame)

    evento = {
        "tipo": tipo,
        "hora": ahora.strftime("%H:%M:%S"),
        "fecha": ahora.strftime("%d/%m/%Y"),
        "track_id": int(track_id),
        "archivo": nombre_archivo
    }

    with lock:

        ultimos_eventos.insert(0, evento)

        # Mantener solo los últimos 20 eventos en memoria
        if len(ultimos_eventos) > 20:
            ultimos_eventos.pop()

    print(
        f"[{tipo}] "
        f"{evento['fecha']} "
        f"{evento['hora']} "
        f"ID={track_id}"
    )


# ============================================================
# PROCESAMIENTO DE CÁMARA
# ============================================================

def procesar_camara():

    global ultimo_frame
    global contador_ingresos
    global contador_egresos

    print("")
    print("Sistema de detección activo.")
    print("")

    while True:

        success, frame = cap.read()

        if not success:
            print("No se pudo leer el fotograma.")
            continue

        # ----------------------------------------------------
        # YOLO TRACKING
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            classes=[0],       # persona
            conf=0.3,
            verbose=False
        )

        # ----------------------------------------------------
        # LÍNEA DE CRUCE
        # ----------------------------------------------------

        cv2.line(
            frame,
            (X_INICIO, Y_LINEA),
            (X_FIN, Y_LINEA),
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            "LINEA DE ACCESO",
            (X_INICIO, Y_LINEA - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

        # ----------------------------------------------------
        # PERSONAS
        # ----------------------------------------------------

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

                # Centro inferior de la persona
                cx = int((x1 + x2) / 2)
                cy = y2

                # Dibujar persona
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (255, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # DETECTAR CRUCE
                # ------------------------------------------------

                if track_id in posiciones_previas:

                    y_anterior = posiciones_previas[track_id]

                    # ============================================
                    # INGRESO
                    # arriba -> abajo
                    # ============================================

                    if (
                        y_anterior < Y_LINEA - MARGEN
                        and cy >= Y_LINEA + MARGEN
                        and X_INICIO <= cx <= X_FIN
                    ):

                        contador_ingresos += 1

                        registrar_evento(
                            "INGRESO",
                            track_id,
                            frame
                        )

                    # ============================================
                    # EGRESO
                    # abajo -> arriba
                    # ============================================

                    elif (
                        y_anterior > Y_LINEA + MARGEN
                        and cy <= Y_LINEA - MARGEN
                        and X_INICIO <= cx <= X_FIN
                    ):

                        contador_egresos += 1

                        registrar_evento(
                            "EGRESO",
                            track_id,
                            frame
                        )

                posiciones_previas[track_id] = cy

        # ----------------------------------------------------
        # INFORMACIÓN SOBRE EL VÍDEO
        # ----------------------------------------------------

        dentro = contador_ingresos - contador_egresos

        if dentro < 0:
            dentro = 0

        cv2.putText(
            frame,
            f"Ingresos: {contador_ingresos}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Egresos: {contador_egresos}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2
        )

        cv2.putText(
            frame,
            f"Dentro: {dentro}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # Convertir frame a JPG
        ret, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )

        if ret:

            with lock:
                ultimo_frame = buffer.tobytes()


# ============================================================
# GENERADOR DE VIDEO
# ============================================================

def generar_video():

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


# ============================================================
# WEB
# ============================================================

PAGINA = """
<!DOCTYPE html>

<html lang="es">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>Control de acceso - Herbario BAB</title>

<style>

body {
    margin: 0;
    background: #101713;
    color: white;
    font-family: Arial, sans-serif;
}

header {
    background: #183c29;
    padding: 15px;
    text-align: center;
}

header h1 {
    margin: 5px;
    font-size: 25px;
}

.contenedor {
    max-width: 1100px;
    margin: auto;
    padding: 15px;
}

.tarjetas {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 15px;
}

.tarjeta {
    background: #1c2922;
    padding: 15px;
    text-align: center;
    border-radius: 10px;
}

.numero {
    font-size: 35px;
    font-weight: bold;
}

.ingreso {
    color: #51db7a;
}

.egreso {
    color: #ffae42;
}

.dentro {
    color: #62c5ff;
}

.video {
    text-align: center;
}

.video img {
    width: 100%;
    max-width: 900px;
    border-radius: 10px;
    border: 2px solid #365845;
}

.eventos {
    margin-top: 20px;
    background: #1c2922;
    border-radius: 10px;
    padding: 15px;
}

.evento {
    border-bottom: 1px solid #405047;
    padding: 10px;
}

@media (max-width: 600px) {

    .tarjetas {
        grid-template-columns: 1fr;
    }

    header h1 {
        font-size: 19px;
    }

}

</style>

</head>

<body>

<header>

<h1>Control de acceso</h1>

<div>Herbario BAB</div>

</header>


<div class="contenedor">

<div class="tarjetas">

<div class="tarjeta">

<div>INGRESOS</div>

<div
    id="ingresos"
    class="numero ingreso">
0
</div>

</div>


<div class="tarjeta">

<div>EGRESOS</div>

<div
    id="egresos"
    class="numero egreso">
0
</div>

</div>


<div class="tarjeta">

<div>PERSONAS DENTRO</div>

<div
    id="dentro"
    class="numero dentro">
0
</div>

</div>

</div>


<div class="video">

<img src="/video">

</div>


<div class="eventos">

<h2>Últimos movimientos</h2>

<div id="lista_eventos">

Sin movimientos registrados.

</div>

</div>

</div>


<script>

async function actualizarDatos() {

    try {

        const respuesta =
            await fetch("/estado");

        const datos =
            await respuesta.json();

        document.getElementById(
            "ingresos"
        ).innerText =
            datos.ingresos;

        document.getElementById(
            "egresos"
        ).innerText =
            datos.egresos;

        document.getElementById(
            "dentro"
        ).innerText =
            datos.dentro;

        let html = "";

        datos.eventos.forEach(
            function(evento) {

                let color =
                    evento.tipo === "INGRESO"
                    ? "#51db7a"
                    : "#ffae42";

                html +=
                    '<div class="evento">' +
                    '<b style="color:' +
                    color +
                    '">' +
                    evento.tipo +
                    '</b> &nbsp; ' +
                    evento.fecha +
                    ' &nbsp; ' +
                    evento.hora +
                    '</div>';
            }
        );

        if (html === "") {
            html =
                "Sin movimientos registrados.";
        }

        document.getElementById(
            "lista_eventos"
        ).innerHTML = html;

    } catch (error) {

        console.log(error);

    }

}

setInterval(
    actualizarDatos,
    1000
);

actualizarDatos();

</script>

</body>

</html>
"""


# ============================================================
# RUTAS FLASK
# ============================================================

@app.route("/")
def index():

    return render_template_string(PAGINA)


@app.route("/video")
def video():

    return Response(
        generar_video(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


@app.route("/estado")
def estado():

    dentro = contador_ingresos - contador_egresos

    if dentro < 0:
        dentro = 0

    with lock:
        eventos = list(ultimos_eventos)

    return jsonify({
        "ingresos": contador_ingresos,
        "egresos": contador_egresos,
        "dentro": dentro,
        "eventos": eventos
    })


# ============================================================
# INICIO
# ============================================================

if __name__ == "__main__":

    hilo = threading.Thread(
        target=procesar_camara,
        daemon=True
    )

    hilo.start()

    print("")
    print("======================================")
    print(" CONTROL DE ACCESO - HERBARIO BAB")
    print("======================================")
    print("")
    print("Servidor web iniciado.")
    print("")
    print("Desde otra PC o celular usar:")
    print("")
    print("http://IP-DE-LA-NOTEBOOK:5000")
    print("")

    app.run(
        host="0.0.0.0",
        port=PUERTO,
        debug=False,
        threaded=True,
        use_reloader=False
    )