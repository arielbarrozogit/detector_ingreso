import cv2
from ultralytics import YOLO

# Cargar el modelo YOLO (el modelo 'n' -nano- es liviano y corre rápido en CPU)
print("Cargando modelo YOLO...")
model = YOLO("yolov8n.pt")

# Abrir la cámara web de la notebook (0 suele ser la cámara integrada)
cap = cv2.VideoCapture(0)

# Verificar si la cámara abrió correctamente
if not cap.isOpened():
  print("Error: No se pudo acceder a la cámara de la PC.")
  exit()

print("Cámara iniciada. Presiona la tecla 'q' para salir.")

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("No se pudo leer el fotograma de la cámara.")
    break

  # Ejecutar la detección de objetos filtrando únicamente la clase 0 (persona en COCO)
  results = model(frame, classes=[0], conf=0.5, verbose=False)

  # Dibujar las detecciones sobre el fotograma
  annotated_frame = results[0].plot()

  # Contar cuántas personas hay en el fotograma actual
  cantidad_personas = len(results[0].boxes)

  # Mostrar un contador visual en la esquina superior izquierda
  cv2.putText(
      annotated_frame,
      f"Personas en escena: {cantidad_personas}",
      (20, 40),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0, 255, 0),
      2,
  )

  # Mostrar la ventana con el video en tiempo real
  cv2.imshow(
      "PoC Herbario - Deteccion de Ingreso",
      annotated_frame,
  )

  # Salir al presionar la tecla 'q'
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# Liberar los recursos de la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()