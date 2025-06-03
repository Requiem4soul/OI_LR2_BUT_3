import matplotlib.pyplot as plt
import numpy as np

# Данные
images = ['test2_1.jpg', 'test2_2.jpg', 'test2_3.jpg', 'test2_4.jpg']
psnr_before = [27.98, 34.46, 31.66, 26.64]
psnr_after = [32.62, 35.73, 33.47, 28.70]

# Настройка ширины столбцов и позиций
bar_width = 0.35
index = np.arange(len(images))

# Создание диаграммы
plt.figure(figsize=(10, 6))
bars_before = plt.bar(index, psnr_before, bar_width, label='PSNR до обработки', color='blue')
bars_after = plt.bar(index + bar_width, psnr_after, bar_width, label='PSNR после обработки', color='red')

# Добавление подписей значений PSNR над столбцами
for bar in bars_before:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.2f}', ha='center', va='bottom')

for bar in bars_after:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.2f}', ha='center', va='bottom')

# Настройка осей и заголовка
plt.xlabel('Изображения')
plt.ylabel('PSNR (dB)')
plt.title('Сравнение PSNR до и после обработки изображений')
plt.xticks(index + bar_width / 2, images, rotation=45)
plt.legend()

# Отображение диаграммы
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()