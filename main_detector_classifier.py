from pathlib import Path #Nuevamente para las rutas
from argparse import ArgumentParser #Esta función ayuda con los argumentos de la linea de comandos
from PytorchWildlife.models import detection as pw_detection #Aquí obtenemos los modelos de detección
from PytorchWildlife.models import classification as pw_classification #Aquí los modelos de clasificación
from utils import process_image, process_folder #Las funciones en común

#De la pagina de Megadetector Pyhtorch Wildlife
#Se obtuvieron los siguientes detectores 
#(Al menos los que sí pude cargar)
DETECTORS = {
    "YOLOv9_Compact": "MDV6-yolov9-c",
    "YOLOv9_Extra": "MDV6-yolov9-e",
    "YOLOv10_Compact": "MDV6-yolov10-c",
    "YOLOv10_Extra": "MDV6-yolov10-e",
    "RtDetr_Compact": "MDV6-rtdetr-c"
}
#De la misma página, se obtuvieron los siguientes detectores
#Los que sí cargaron

CLASSIFIERS = {
    "AI4G Amazon Rainforest": "AI4GAmazonRainforest",
    "AI4G Snapshot Serengeti": "AI4GSnapshotSerengeti",
    "AI4G Opossum": "AI4GOpossum",
    "AI4G Central Africa": "AI4GCentralAfrica"
}

#Se desplegará una lista con las opciones de clasificador y de detector, para ello, definamos la función
def choose_from_menu(options_dict, title):
    print(f"\n{title}:")
    for i, key in enumerate(options_dict.keys(), start=1):
        print(f"{i}. {key}")
    choice = int(input("Selecciona el número deseado: ")) - 1
    key = list(options_dict.keys())[choice]
    return key, options_dict[key]

#Para obtener el indice del detector solicitado
#Lo que importa:
#La función en general pedirá:
#Un detector, un clsificador, una ruta de entrada (o imagen) y el margen para recortar.
if __name__ == "__main__":
    parser = ArgumentParser(
        prog="main_detector_classifier",
        description="Detecta y clasifica animales en imágenes o carpetas, permitiendo elegir modelos."
    )
    parser.add_argument("path", type=Path, help="Ruta de la imagen o carpeta a procesar")
    parser.add_argument("--margin", type=int, default=5, help="Tamaño del margen alrededor de la detección")

    args = parser.parse_args()

    # Se esperá que podamos escoger entre los distintos detectores y clasificadores, entonces, haremos un menú interactivo
#Este menú se desplegará cuando se inicié el programa
    det_name, det_version = choose_from_menu(DETECTORS, "Detectores disponibles")
    clf_name, clf_class = choose_from_menu(CLASSIFIERS, "Clasificadores disponibles")

    # Cargaremos los modelos y se imprimirá la elección
    print(f"\nCargando detector: {det_name}")
    detector = pw_detection.MegaDetectorV6(version=det_version)

    print(f"Cargando clasificador: {clf_name}")
    classifier = getattr(pw_classification, clf_class)()

    # Usaremos una función diferente dependiendo si 
#Se clasificará una imagen o todo un bonche de imgenes:
#Si el argumento es una carpeta:
    if args.path.is_dir():
        process_folder(args.path, detector, classifier, args.margin)
#Si el argumento es una imagen
    elif args.path.is_file():
        output_folder = args.path.parent / "recortes_single"
        output_folder.mkdir(exist_ok=True)
        process_image(args.path, output_folder, detector, classifier, args.margin)
#Si no es ninguna de las dos 
    else:
        print("argumento de entrada no válido")
