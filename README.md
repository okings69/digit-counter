# Digit Counter

This project counts how many times each digit (0–9) appears in a dataset of grayscale images.

## Result
[1246, 1547, 950, 1264, 1269, 1275, 1226, 777, 968, 1478]

## Approach
- Used MNIST dataset for training
- Preprocessed images to match MNIST format
- Trained multiple small neural networks (NumPy)
- Used majority voting for stability
- Counted final predictions

## How to run

```bash
python solve_digits.py --images-dir digits
