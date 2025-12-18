from pathlib import Path 
from typing import Any
import numpy as np 
from PIL import Image 


def crop_with_margin(image: np.ndarray, box: np.ndarray, margin: int = 5) -> np.ndarray:
    """ Recorta imágenes por un recuadro aumentando un margen.

    Args:
        image (np.ndarray): Array de la imagen.
        box (np.ndarray[int]): Coordenadas de la caja para el recorte.
        margin (int): tamaño en Pixeles del margen alredor de la caja. Default: 5.

    Returns:
        np.ndarray: Array de la imagen recortada
    """
    x1, y1, x2, y2 = box.astype(int)
    height, width, _ = image.shape
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(width, x2 + margin)
    y2 = min(height, y2 + margin)
    return image[y1:y2, x1:x2]

def process_image(image_path: Path, output_folder: Path, detector: Any, classifier: Any, margin: int = 5) -> None:
    """
    Procesa una imagen individual: detecta objetos, los recorta y clasifica su especie. 
	
    Args:
        image_path (Path): Ruta al archivo de la imagen original.
        output_folder (Path): Carpeta donde se guardarán los recortes procesados.
        detector (Any): Modelo de detección (debe tener método .predictor).
        classifier (Any): Modelo del clasificador (debe tener método .single_image_classification).
        margin (int, optional):  Tamaño en pixeles del margen alredor de la caja. Default: 5
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    results = detector.predictor(image)

    for i, result in enumerate(results):
        boxes = result.boxes.xyxy.cpu().numpy()
		
		for j, box in enumerate(boxes):
            # Recortar y clasificar
            cropped = crop_with_margin(result.orig_img, box, margin)
            predicted_species = classifier.single_image_classification(cropped)
            species_name = predicted_species['prediction']

            #Nombre de archivo y guardar
            # Estructura: Original_det[NumDeteccion]_[NumCaja]_[Especie].jpg
            output_name = output_folder / f"{image_path.stem}_det{i+1}_{j+1}_{species_name}.jpg"
            Image.fromarray(cropped).save(output_name)


def process_folder(folder_path: Path, detector: Any, classifier: Any, margin: int = 5) -> None:
    """
    Procesa todas las imágenes dentro de un directorio. Crea una subcarpeta 'recortes' (si no existe) e itera sobre todos los archivos
    con extensión .jpg encontrados aplicando la función process_image.

    Args:
        folder_path (Path): Ruta del directorio que contiene las imágenes.
        detector (Any): Modelo de detección.
        classifier (Any): Modelo de clasificación.
        margin (int, optional):  Tamaño en pixeles del margen alredor de la caja. Default: 5.
    """
    output_folder = folder_path / "recortes"
    output_folder.mkdir(exist_ok=True)

    # Iterar sobre imágenes ordenadas alfabéticamente
    for image_path in sorted(folder_path.glob("*.jpg")):
        process_image(image_path, output_folder, detector, classifier, margin)

