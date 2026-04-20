from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SEEDS = [0, 1, 2, 3, 4]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compte les chiffres 0-9 à partir d'un dossier d'images."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("digits"),
        help="Dossier contenant les images à classifier",
    )
    parser.add_argument(
        "--mnist-path",
        type=Path,
        default=Path("mnist.npz"),
        help="Chemin local du fichier MNIST",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Nombre d'époques d'entraînement par modèle",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Taille des mini-batchs",
    )
    return parser.parse_args()


def download_mnist_if_needed(path: Path) -> Path:
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement de MNIST vers : {path}")
    urllib.request.urlretrieve(MNIST_URL, path)
    return path


def preprocess_image_array(arr: np.ndarray, output_size: int = 28) -> np.ndarray:
    """
    Prépare une image pour qu'elle ressemble davantage au format MNIST :
    - niveaux de gris
    - suppression du bruit faible
    - recadrage autour du chiffre
    - redimensionnement
    - centrage
    - normalisation
    """
    arr = arr.astype(np.uint8)

    # On supprime les pixels très faibles pour nettoyer un peu le fond
    arr = np.where(arr < 20, 0, arr)

    ys, xs = np.where(arr > 0)

    # Cas rare : image vide
    if len(xs) == 0:
        return np.zeros((output_size, output_size), dtype=np.float32)

    # Boîte englobante du chiffre
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cropped = arr[y_min:y_max + 1, x_min:x_max + 1]

    h, w = cropped.shape

    # On fait tenir le chiffre dans une boîte d'environ 20x20,
    # puis on le place dans une image 28x28 comme MNIST
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    resized = Image.fromarray(cropped, mode="L").resize((new_w, new_h), resample)

    canvas = np.zeros((output_size, output_size), dtype=np.float32)
    top = (output_size - new_h) // 2
    left = (output_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = np.array(resized, dtype=np.float32)

    # Recentrage via centre de masse
    total = canvas.sum()
    if total > 0:
        yy, xx = np.indices(canvas.shape)
        center_y = float((yy * canvas).sum() / total)
        center_x = float((xx * canvas).sum() / total)

        shift_y = int(round(output_size / 2 - center_y))
        shift_x = int(round(output_size / 2 - center_x))

        shifted = np.zeros_like(canvas)

        src_y0 = max(0, -shift_y)
        dst_y0 = max(0, shift_y)
        src_x0 = max(0, -shift_x)
        dst_x0 = max(0, shift_x)

        copy_h = min(output_size - dst_y0, output_size - src_y0)
        copy_w = min(output_size - dst_x0, output_size - src_x0)

        if copy_h > 0 and copy_w > 0:
            shifted[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = canvas[
                src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w
            ]
            canvas = shifted

    return canvas / 255.0


def load_target_images(images_dir: Path) -> tuple[list[Path], np.ndarray]:
    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"Aucune image .jpg trouvée dans {images_dir}")

    processed = []
    for path in image_paths:
        img = Image.open(path).convert("L")
        arr = np.array(img)
        processed.append(preprocess_image_array(arr))

    return image_paths, np.stack(processed).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def forward_pass(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z1 = x @ w1 + b1
    a1 = np.maximum(0, z1)  # ReLU

    z2 = a1 @ w2 + b2
    a2 = np.maximum(0, z2)  # ReLU

    logits = a2 @ w3 + b3
    return z1, a1, z2, a2, logits


def train_one_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
) -> np.ndarray:
    np.random.seed(seed)

    input_size = 784
    hidden_1 = 256
    hidden_2 = 128
    output_size = 10

    w1 = np.random.randn(input_size, hidden_1).astype(np.float32) * np.sqrt(2.0 / input_size)
    b1 = np.zeros(hidden_1, dtype=np.float32)

    w2 = np.random.randn(hidden_1, hidden_2).astype(np.float32) * np.sqrt(2.0 / hidden_1)
    b2 = np.zeros(hidden_2, dtype=np.float32)

    w3 = np.random.randn(hidden_2, output_size).astype(np.float32) * np.sqrt(2.0 / hidden_2)
    b3 = np.zeros(output_size, dtype=np.float32)

    learning_rate = 0.03

    for epoch in range(epochs):
        order = np.random.permutation(len(x_train))

        for start in range(0, len(x_train), batch_size):
            batch_ids = order[start:start + batch_size]
            xb = x_train[batch_ids]
            yb = y_train[batch_ids]

            z1, a1, z2, a2, logits = forward_pass(xb, w1, b1, w2, b2, w3, b3)

            probs = softmax(logits)
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)

            dw3 = a2.T @ probs
            db3 = probs.sum(axis=0)

            da2 = probs @ w3.T
            da2[z2 <= 0] = 0

            dw2 = a1.T @ da2
            db2 = da2.sum(axis=0)

            da1 = da2 @ w2.T
            da1[z1 <= 0] = 0

            dw1 = xb.T @ da1
            db1 = da1.sum(axis=0)

            w3 -= learning_rate * dw3
            b3 -= learning_rate * db3

            w2 -= learning_rate * dw2
            b2 -= learning_rate * db2

            w1 -= learning_rate * dw1
            b1 -= learning_rate * db1

        learning_rate *= 0.9
        print(f"Modèle seed={seed} - époque {epoch + 1}/{epochs} terminée")

    *_, logits_test = forward_pass(x_test, w1, b1, w2, b2, w3, b3)
    return logits_test.argmax(axis=1)


def majority_vote(all_predictions: np.ndarray) -> np.ndarray:
    """
    all_predictions shape = (nombre_modeles, nombre_images)
    Pour chaque image, on compte les votes des modèles.
    """
    votes = np.apply_along_axis(
        lambda col: np.bincount(col, minlength=10),
        axis=0,
        arr=all_predictions,
    )
    return votes.argmax(axis=0)


def main() -> None:
    args = get_args()

    mnist_path = download_mnist_if_needed(args.mnist_path)
    mnist = np.load(mnist_path)

    print("Prétraitement des images MNIST...")
    x_train = np.stack(
        [preprocess_image_array(img) for img in mnist["x_train"]]
    ).reshape(60000, -1)
    y_train = mnist["y_train"]

    # Standardisation
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    x_train = ((x_train - mean) / std).astype(np.float32)

    print("Chargement des images cibles...")
    files, x_target = load_target_images(args.images_dir)
    x_target = x_target.reshape(len(files), -1)
    x_target = ((x_target - mean) / std).astype(np.float32)

    print(f"Nombre d'images à classifier : {len(files)}")

    predictions_per_model = []
    for seed in SEEDS:
        preds = train_one_model(
            x_train=x_train,
            y_train=y_train,
            x_test=x_target,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        predictions_per_model.append(preds)

    predictions_per_model = np.stack(predictions_per_model)
    final_predictions = majority_vote(predictions_per_model)

    counts = np.bincount(final_predictions, minlength=10).tolist()

    print("Résultat final :")
    print(json.dumps(counts))

    # Sauvegarde pratique
    Path("result.json").write_text(json.dumps(counts), encoding="utf-8")
    print("Résultat sauvegardé dans result.json")


if __name__ == "__main__":
    main()