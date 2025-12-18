from pathlib import Path
from argparse import ArgumentParser
from PytorchWildlife.models import detection as pw_detection
from utils import process_detection_only, process_folder_detection_only

# Detectores probados basados en los disponibles en https://microsoft.github.io/CameraTraps/model_zoo/megadetector/
DETECTORS = {
    "YOLOv9_Compact": "MDV6-yolov9-c",
    "YOLOv9_Extra": "MDV6-yolov9-e",
    "YOLOv10_Compact": "MDV6-yolov10-c",
    "YOLOv10_Extra": "MDV6-yolov10-e",
    "RtDetr_Compact": "MDV6-rtdetr-c"
}
DETECTORS_OPTS = list(DETECTORS.keys())

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="detection_tool",
        description="Herramienta exclusiva para detección y recorte de animales, permitiendo elegir el modelo."
    )
    parser.add_argument("path", type=Path, help="Ruta de la imagen o carpeta a procesar")
    parser.add_argument("--detector", default=DETECTORS_OPTS[0], choices=DETECTORS_OPTS, 
                        help="Modelo de detección que se desea usar")
    parser.add_argument("--margin", type=int, default=5, help="Tamaño del margen alrededor de la detección")

    args = parser.parse_args()

    # Cargaremos el modelo y se imprimirá la elección
    print(f"\nCargando detector: {args.detector}")
    detector = pw_detection.MegaDetectorV6(version=DETECTORS[args.detector])

    # Si path es directorio se procesan todas las imagenes en el directorio
    if args.path.is_dir():
        # process_folder_detection_only ya se encarga de crear la carpeta de salida internamente
        process_folder_detection_only(args.path, detector, args.margin)
    elif args.path.is_file():
        output_folder = args.path.parent / "recortes_sin_clasificar_single"
        output_folder.mkdir(exist_ok=True)
        process_detection_only(args.path, output_folder, detector, args.margin)
    else:
        raise ValueError(f"{args.path} no es un valor válido")
