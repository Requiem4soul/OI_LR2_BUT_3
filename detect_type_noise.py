import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Пути к изображениям
base_path = "Data"
reference_image_path = os.path.join(base_path, "test2_0.jpg")
noisy_images = [os.path.join(base_path, f"test2_{i}.jpg") for i in range(1, 5)]

# Загрузка эталона и вычисление гистограммы
ref_img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
ref_hist = cv2.calcHist([ref_img], [0], None, [256], [0, 256]).flatten()

# Нормировка гистограммы для сравнения
ref_hist_norm = ref_hist / ref_hist.sum()

for noisy_path in noisy_images:
    noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
    noisy_hist = cv2.calcHist([noisy_img], [0], None, [256], [0, 256]).flatten()
    noisy_hist_norm = noisy_hist / noisy_hist.sum()

    # Разностная гистограмма
    hist_diff = noisy_hist_norm - ref_hist_norm

    # Построение графиков
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(ref_hist_norm, label='Эталон')
    plt.plot(noisy_hist_norm, label='С шумом')
    plt.title(f'Гистограмма: {os.path.basename(noisy_path)}')
    plt.xlabel('Яркость (0-255)')
    plt.ylabel('Норм. частота')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(hist_diff, color='red')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('Разность гистограмм (Шумное - Эталон)')
    plt.xlabel('Яркость (0-255)')
    plt.ylabel('Разница')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
