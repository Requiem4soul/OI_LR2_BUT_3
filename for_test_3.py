import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Пути
base_path = "Data"
ref_path = os.path.join(base_path, "test2_0.jpg")
img3_path = os.path.join(base_path, "test2_3.jpg")

# Координаты обрезки
x_start, x_end = 150, 450
y_start, y_end = 200, 400

# Загрузка и обрезка изображений
ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread(img3_path, cv2.IMREAD_GRAYSCALE)

ref_crop = ref_img[y_start:y_end, x_start:x_end]
img3_crop = img3[y_start:y_end, x_start:x_end]

# Сохранение обрезков (для наглядности)
cv2.imwrite(os.path.join(base_path, "test2_00.jpg"), ref_crop)
cv2.imwrite(os.path.join(base_path, "test2_33.jpg"), img3_crop)

# Гистограммы
ref_hist = cv2.calcHist([ref_crop], [0], None, [256], [0, 256]).flatten()
img3_hist = cv2.calcHist([img3_crop], [0], None, [256], [0, 256]).flatten()

ref_hist_norm = ref_hist / ref_hist.sum()
img3_hist_norm = img3_hist / img3_hist.sum()
diff_hist = img3_hist_norm - ref_hist_norm

# Графики
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(ref_hist_norm, label="Эталон (вырез)")
plt.plot(img3_hist_norm, label="Шумное (вырез)")
plt.title("Гистограммы обрезанных областей")
plt.xlabel("Яркость (0-255)")
plt.ylabel("Норм. частота")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(diff_hist, color='red')
plt.axhline(0, color='black', linewidth=0.5)
plt.title("Разность гистограмм")
plt.xlabel("Яркость (0-255)")
plt.ylabel("Разница")
plt.grid(True)

plt.tight_layout()
plt.show()
