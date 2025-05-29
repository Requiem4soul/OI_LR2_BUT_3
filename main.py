import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from skimage import filters, restoration
from skimage.metrics import peak_signal_noise_ratio
import seaborn as sns


class NoiseAnalyzer:
    def __init__(self):
        self.noise_types = {
            'gaussian': 'Гауссовский шум',
            'salt_pepper': 'Импульсный шум (соль и перец)',
            'uniform': 'Равномерный шум',
            'poisson': 'Пуассоновский шум',
            'speckle': 'Мультипликативный шум'
        }

    def load_images(self, paths):
        """Загрузка изображений"""
        images = {}
        for name, path in paths.items():
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images[name] = img.astype(np.float64) / 255.0
            else:
                print(f"Не удалось загрузить изображение: {path}")
        return images

    def calculate_psnr(self, original, noisy):
        """Расчет PSNR между изображениями"""
        if original.shape != noisy.shape:
            print("Изображения должны иметь одинаковые размеры")
            return None
        return peak_signal_noise_ratio(original, noisy, data_range=1.0)

    def evaluate_all_filters(self, image, original=None):
        """Применяет все фильтры и сравнивает их (если есть эталон)"""
        filter_variants = {
            'Гауссовский (σ=1.0)': self.apply_gaussian_filter(image, sigma=1.0),
            'Гауссовский (σ=1.5)': self.apply_gaussian_filter(image, sigma=1.5),
            'Медианный (3x3)': self.apply_median_filter(image, disk_size=3),
            'Медианный (5x5)': self.apply_median_filter(image, disk_size=5),
            'Винера': self.apply_wiener_filter(image, noise_variance=0.01),
            'Билатеральный': self.apply_bilateral_filter(image)
        }

        psnr_results = {}
        if original is not None:
            for name, filtered in filter_variants.items():
                psnr = self.calculate_psnr(original, filtered)
                psnr_results[name] = psnr

        return filter_variants, psnr_results

    def analyze_histogram(self, noise_region):
        """Анализ гистограммы для определения типа шума"""
        hist, bins = np.histogram(noise_region.flatten(), bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Статистические характеристики
        mean_val = np.mean(noise_region)
        std_val = np.std(noise_region)
        skewness = stats.skew(noise_region.flatten())
        kurtosis = stats.kurtosis(noise_region.flatten())

        # Тесты на нормальность распределения
        shapiro_stat, shapiro_p = stats.shapiro(noise_region.flatten()[:5000])  # ограничиваем выборку

        analysis = {
            'mean': mean_val,
            'std': std_val,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'histogram': (hist, bin_centers)
        }

        return analysis

    def determine_noise_type(self, analysis):
        """Определение типа шума на основе статистического анализа"""
        skewness = analysis['skewness']
        kurtosis = analysis['kurtosis']
        shapiro_p = analysis['shapiro_p']

        # Критерии для определения типа шума
        if shapiro_p > 0.05:  # Гауссовское распределение
            if abs(skewness) < 0.5 and abs(kurtosis) < 3:
                return 'gaussian', f"Гауссовский шум (p-value={shapiro_p:.4f}, асимметрия={skewness:.3f})"

        if kurtosis > 10:  # Высокий эксцесс - признак импульсного шума
            return 'salt_pepper', f"Импульсный шум (эксцесс={kurtosis:.3f})"

        if abs(skewness) < 0.3 and 2 < kurtosis < 5:
            return 'uniform', f"Равномерный шум (асимметрия={skewness:.3f}, эксцесс={kurtosis:.3f})"

        return 'gaussian', f"Предположительно гауссовский шум (требует дополнительного анализа)"

    def extract_noise_region(self, image, region_size=50):
        """Извлечение однородной области для анализа шума"""
        h, w = image.shape
        # Берем область из центра изображения (обычно менее детализированная)
        center_y, center_x = h // 2, w // 2
        y1 = max(0, center_y - region_size // 2)
        y2 = min(h, center_y + region_size // 2)
        x1 = max(0, center_x - region_size // 2)
        x2 = min(w, center_x + region_size // 2)

        return image[y1:y2, x1:x2]

    def apply_gaussian_filter(self, image, sigma=1.0):
        """Применение Гауссовского фильтра"""
        return filters.gaussian(image, sigma=sigma)

    def apply_median_filter(self, image, disk_size=3):
        """Применение медианного фильтра"""
        return filters.median(image, np.ones((disk_size, disk_size)))

    def apply_wiener_filter(self, image, noise_variance=0.01):
        """Применение фильтра Винера"""
        # Простая реализация фильтра Винера
        return restoration.wiener(image, np.ones((5, 5)) / 25, noise_variance)

    def apply_bilateral_filter(self, image, sigma_color=0.1, sigma_spatial=15):
        """Применение билатерального фильтра"""
        # Преобразуем в uint8 для работы с OpenCV
        img_uint8 = (image * 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(img_uint8, -1, sigma_color * 255, sigma_spatial)
        return filtered.astype(np.float64) / 255.0

    def select_optimal_filter(self, noise_type, image, original=None):
        """Выбор оптимального фильтра на основе типа шума"""
        filters_to_test = {}

        if noise_type == 'gaussian':
            filters_to_test['Гауссовский (σ=0.8)'] = self.apply_gaussian_filter(image, sigma=0.8)
            filters_to_test['Гауссовский (σ=1.2)'] = self.apply_gaussian_filter(image, sigma=1.2)
            filters_to_test['Винера'] = self.apply_wiener_filter(image, noise_variance=0.01)
            filters_to_test['Билатеральный'] = self.apply_bilateral_filter(image)

        elif noise_type == 'salt_pepper':
            filters_to_test['Медианный (3x3)'] = self.apply_median_filter(image, disk_size=3)
            filters_to_test['Медианный (5x5)'] = self.apply_median_filter(image, disk_size=5)
            filters_to_test['Билатеральный'] = self.apply_bilateral_filter(image)

        else:  # универсальный подход
            filters_to_test['Гауссовский'] = self.apply_gaussian_filter(image, sigma=1.0)
            filters_to_test['Медианный'] = self.apply_median_filter(image, disk_size=3)
            filters_to_test['Билатеральный'] = self.apply_bilateral_filter(image)

        # Если есть оригинальное изображение, выбираем лучший по PSNR
        if original is not None:
            best_filter = None
            best_psnr = 0
            psnr_results = {}

            for filter_name, filtered_img in filters_to_test.items():
                psnr = self.calculate_psnr(original, filtered_img)
                psnr_results[filter_name] = psnr
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_filter = filter_name

            return filters_to_test[best_filter], best_filter, psnr_results

        # Если нет оригинала, возвращаем первый фильтр
        first_filter_name = list(filters_to_test.keys())[0]
        return filters_to_test[first_filter_name], first_filter_name, {}

    def plot_histogram_analysis(self, image, noise_region, analysis, noise_type_info):
        """Визуализация анализа гистограммы"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Исходное изображение
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Изображение с шумом')
        axes[0, 0].axis('off')

        # Область для анализа шума
        axes[0, 1].imshow(noise_region, cmap='gray')
        axes[0, 1].set_title('Область анализа шума')
        axes[0, 1].axis('off')

        # Гистограмма
        hist, bin_centers = analysis['histogram']
        axes[1, 0].bar(bin_centers, hist, width=bin_centers[1] - bin_centers[0], alpha=0.7)
        axes[1, 0].set_title('Гистограмма области шума')
        axes[1, 0].set_xlabel('Значение интенсивности')
        axes[1, 0].set_ylabel('Плотность')

        # Статистики
        stats_text = f"""Статистический анализ:
Среднее: {analysis['mean']:.4f}
Ст. отклонение: {analysis['std']:.4f}
Асимметрия: {analysis['skewness']:.4f}
Эксцесс: {analysis['kurtosis']:.4f}
Тест Шапиро-Уилка: {analysis['shapiro_p']:.4f}

Тип шума: {noise_type_info}"""

        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='center')
        axes[1, 1].axis('off')

        plt.tight_layout()
        return fig

    def analyze_single_image(self, image_name, image, original=None):
        """Полный анализ одного изображения"""
        print(f"\n=== Анализ изображения: {image_name} ===")

        # Извлекаем область для анализа шума
        noise_region = self.extract_noise_region(image)

        # Анализируем гистограмму
        analysis = self.analyze_histogram(noise_region)

        # Определяем тип шума
        noise_type, noise_explanation = self.determine_noise_type(analysis)

        print(f"Тип шума: {self.noise_types[noise_type]}")
        print(f"Обоснование: {noise_explanation}")

        # Выбираем и применяем фильтр
        filtered_image, filter_name, psnr_results = self.select_optimal_filter(
            noise_type, image, original)

        # Расчет PSNR
        if original is not None:
            psnr_before = self.calculate_psnr(original, image)
            psnr_after = self.calculate_psnr(original, filtered_image)
            print(f"PSNR до фильтрации: {psnr_before:.2f} дБ")
            print(f"PSNR после фильтрации ({filter_name}): {psnr_after:.2f} дБ")
            print(f"Улучшение: {psnr_after - psnr_before:.2f} дБ")

        # Обоснование выбора метода
        filter_justification = self.get_filter_justification(noise_type, filter_name)
        print(f"Обоснование метода: {filter_justification}")

        # Создаем визуализацию
        fig = self.plot_histogram_analysis(image, noise_region, analysis, noise_explanation)

        all_filtered_images, all_psnrs = self.evaluate_all_filters(image, original)
        print("=== Оценка всех фильтров ===")
        for fname, psnr_val in all_psnrs.items():
            print(f"{fname}: {psnr_val:.2f} дБ")

        return {
            'noise_type': self.noise_types[noise_type],
            'noise_explanation': noise_explanation,
            'filter_name': filter_name,
            'filter_justification': filter_justification,
            'filtered_image': filtered_image,
            'psnr_before': self.calculate_psnr(original, image) if original is not None else None,
            'psnr_after': self.calculate_psnr(original, filtered_image) if original is not None else None,
            'psnr_results': psnr_results,
            'analysis_plot': fig
        }

    def get_filter_justification(self, noise_type, filter_name):
        """Обоснование выбора фильтра"""
        justifications = {
            'gaussian': {
                'Гауссовский': "Гауссовский фильтр эффективен для гауссовского шума, так как выполняет линейное сглаживание с весами по нормальному распределению",
                'Винера': "Фильтр Винера оптимален для гауссовского шума, минимизирует среднеквадратичную ошибку",
                'Билатеральный': "Билатеральный фильтр сохраняет края при подавлении гауссовского шума"
            },
            'salt_pepper': {
                'Медианный': "Медианный фильтр эффективно устраняет импульсный шум, заменяя выбросы медианным значением окружения",
                'Билатеральный': "Билатеральный фильтр может подавлять импульсный шум с сохранением краев"
            }
        }
        return justifications.get(noise_type, {}).get(filter_name.split('(')[0].strip(),
                                                      "Универсальный подход для данного типа шума")


def main():
    """Основная функция для выполнения лабораторной работы"""
    analyzer = NoiseAnalyzer()

    # Пути к изображениям (замените на свои)
    image_paths = {
    'test2_0': r'OI_LR2_BUT_3\Data\test2_0.jpg',
    'test2_1': r'OI_LR2_BUT_3\Data\test2_1.jpg',
    'test2_2': r'OI_LR2_BUT_3\Data\test2_2.jpg',
    'test2_3': r'OI_LR2_BUT_3\Data\test2_3.jpg',
    'test2_4': r'OI_LR2_BUT_3\Data\test2_4.jpg'
    }

    # Загружаем изображения
    images = analyzer.load_images(image_paths)

    if 'test2_0' not in images:
        print("Оригинальное изображение не найдено!")
        return

    original = images['test2_0']
    results = {}

    # Анализируем каждое изображение с шумом
    for img_name in ['test2_1', 'test2_2', 'test2_3', 'test2_4']:
        if img_name in images:
            result = analyzer.analyze_single_image(img_name, images[img_name], original)
            results[img_name] = result

    # Создаем сводную диаграмму PSNR
    if results:
        create_psnr_comparison(results)

    # Выводим таблицу результатов
    print_results_table(results)

    plt.show()


def create_psnr_comparison(results):
    """Создание диаграммы сравнения PSNR"""
    fig, ax = plt.subplots(figsize=(12, 6))

    images = list(results.keys())
    psnr_before = [results[img]['psnr_before'] for img in images if results[img]['psnr_before']]
    psnr_after = [results[img]['psnr_after'] for img in images if results[img]['psnr_after']]

    x = np.arange(len(images))
    width = 0.35

    ax.bar(x - width / 2, psnr_before, width, label='До фильтрации', alpha=0.7)
    ax.bar(x + width / 2, psnr_after, width, label='После фильтрации', alpha=0.7)

    ax.set_xlabel('Изображения')
    ax.set_ylabel('PSNR (дБ)')
    ax.set_title('Сравнение PSNR до и после фильтрации')
    ax.set_xticks(x)
    ax.set_xticklabels(images)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()


def print_results_table(results):
    """Вывод таблицы результатов"""
    print("\n" + "=" * 100)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 100)

    headers = ["Изображение", "Тип шума", "Метод фильтрации", "PSNR до", "PSNR после", "Улучшение"]
    print(
        f"{headers[0]:<12} | {headers[1]:<20} | {headers[2]:<20} | {headers[3]:<8} | {headers[4]:<10} | {headers[5]:<10}")
    print("-" * 100)

    for img_name, result in results.items():
        psnr_before = f"{result['psnr_before']:.2f}" if result['psnr_before'] else "N/A"
        psnr_after = f"{result['psnr_after']:.2f}" if result['psnr_after'] else "N/A"
        improvement = f"{result['psnr_after'] - result['psnr_before']:.2f}" if result['psnr_before'] and result[
            'psnr_after'] else "N/A"

        print(
            f"{img_name:<12} | {result['noise_type']:<20} | {result['filter_name']:<20} | {psnr_before:<8} | {psnr_after:<10} | {improvement:<10}")


if __name__ == "__main__":
    main()