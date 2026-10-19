import os
import cv2
from ultralytics import YOLO

# Crear una carpeta para almacenar las capturas si no existe
os.makedirs("capturas_ingresos", exist_ok=True)

# Cargar el modelo YOLO
print("Cargando modelo YOLO...")
model = YOLO("yolov8n.pt")

# Abrir la cámara web forzando el backend de Windows para evitar errores
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
  print("Error: No se pudo abrir la cámara.")
  exit()

# Configurar la línea virtual (Coordenada Y fija y rango horizontal X)
y_linea = 300
x_inicio, x_fin = 100, 540

# Control de IDs para evitar guardar varias fotos de la misma persona
ids_cruzados = set()
posiciones_previas = {}
contador_ingresos = 0

print("Sistema activo. Presiona 'q' para salir.")

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("No se pudo leer el fotograma.")
    break

  # Usar .track() para mantener los IDs únicos de cada persona
  results = model.track(
      frame, persist=True, classes=[0], conf=0.3, verbose=False
  )

  # Dibujar la línea virtual de cruce en color rojo
  cv2.line(frame, (x_inicio, y_linea), (x_fin, y_linea), (0, 0, 255), 3)

  # Verificar si se detectaron personas
  if results[0].boxes.id is not None:
    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.cpu().numpy().astype(int)

    for box, track_id in zip(boxes, track_ids):
      x1, y1, x2, y2 = map(int, box)
      
      # Centro inferior del cuadro (base del cuerpo / pies)
      cx = int((x1 + x2) / 2)
      cy = y2

      # Dibujar la caja y el ID en pantalla
      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(
          frame,
          f"ID: {track_id}",
          (x1, y1 - 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 255, 0),
          2,
      )

      # Lógica de cruce de línea
      if track_id in posiciones_previas:
        y_anterior = posiciones_previas[track_id]

        # Si estaba arriba y pasó abajo de la línea
        if y_anterior < y_linea and cy >= y_linea and track_id not in ids_cruzados:
          if x_inicio <= cx <= x_fin:
            contador_ingresos += 1
            ids_cruzados.add(track_id)

            # Guardar la captura
            nombre_archivo = f"capturas_ingresos/ingreso_id_{track_id}.jpg"
            cv2.imwrite(nombre_archivo, frame)
            print(f"[ACCESO REGISTRADO] Foto guardada: {nombre_archivo}")

      posiciones_previas[track_id] = cy

  # Mostrar el contador total en la pantalla
  cv2.putText(
      frame,
      f"Ingresos Totales: {contador_ingresos}",
      (30, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (255, 0, 0),
      2,
  )

  # ** ESTO ES CLAVE: Mostrar la ventana de video **
  cv2.imshow("Control de Ingreso - Herbario", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()