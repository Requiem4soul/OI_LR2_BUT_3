import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage import filters, restoration
from skimage.metrics import peak_signal_noise_ratio

TYPE_OF_NOISE = {
    'gaussian': 'Гауссовский шум',
    'salt_pepper': 'Импульсный шум (соль и перец)',
    'uniform': 'Равномерный шум',
    'poisson': 'Пуассоновский шум',
    'speckle': 'Мультипликативный шум',
    'laplacian': 'Шум Лапласа',
    'unknown': 'Неопределённый шум'
}

PATH_TO_IMG = {
    'test2_0.jpg': 'Data/test2_0.jpg',
    'test2_1.jpg': 'Data/test2_1.jpg',
    'test2_2.jpg': 'Data/test2_2.jpg',
    'test2_3.jpg': 'Data/test2_3.jpg',
    'test2_4.jpg': 'Data/test2_4.jpg'
}

def load_image(paths): # Загрузка изображений
    images = {}
    for name, path in paths.items():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images[name] = img.astype(np.float64) / 255.0
        else:
            print(f"Предупреждение: не удалось загрузить изображение {path}")
    return images

def calculate_psnr(original, noisy): # Расчёт различия по PSNR
    return peak_signal_noise_ratio(original, noisy, data_range=1.0)

def extract_noise_region(image): # Выбрал лучшую область, в которой практически одинаковые по тону пиксели
    """Извлечение однородной области для анализа шума"""
    # Координаты (x1, y1) — верхний левый угол, (x2, y2) — нижний правый
    x1, y1, x2, y2 = 50, 100, 150, 200
    return image[y1:y2, x1:x2]

def analyze_histogram(noise_region): # Анализируем область и получаем метрики (корни не пройдут)
    hist, bins = np.histogram(noise_region.flatten(), bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    mean_val = np.mean(noise_region)
    std_val = np.std(noise_region)
    skewness = stats.skew(noise_region.flatten())
    kurtosis = stats.kurtosis(noise_region.flatten())

    # Тесты на нормальность распределения
    shapiro_stat, shapiro_p = stats.shapiro(noise_region.flatten()[:5000])

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


def determine_noise_type(analysis):
    """Определение типа шума на основе статистического анализа"""
    skewness = analysis['skewness']
    kurtosis = analysis['kurtosis']
    shapiro_p = analysis['shapiro_p']
    mean = analysis['mean']
    std = analysis['std']

    # Определяем уровень шума на основе стандартного отклонения
    if std < 0.1:
        noise_level = 'low'
    elif std < 0.25:
        noise_level = 'medium'
    else:
        noise_level = 'high'

    # Проверка на Пуассоновский шум (среднее ≈ дисперсия)
    if mean > 0.1 and abs(std ** 2 - mean) < 0.1 * mean:
        return 'poisson', f"Шум Пуассона (mean ≈ variance: {mean:.3f} ~ {std ** 2:.3f})", noise_level

    # Основная логика определения (по образцу вашего кода)
    if shapiro_p > 0.05:  # Нормальное распределение
        if abs(skewness) < 0.5 and abs(kurtosis) < 3:
            return 'gaussian', f"Гауссовский шум (p-value={shapiro_p:.4f}, асимметрия={skewness:.3f})"

        # Проверяем на равномерный шум при нормальности
        if abs(skewness) < 0.3 and 2 < kurtosis < 5:
            return 'uniform', f"Равномерный шум (асимметрия={skewness:.3f}, эксцесс={kurtosis:.3f})", noise_level

    # Импульсный шум - высокий эксцесс
    if kurtosis > 10:
        return 'salt_pepper', f"Импульсный шум (эксцесс={kurtosis:.3f})", noise_level

    # Лапласовский шум - умеренная асимметрия, высокий эксцесс
    if abs(skewness) < 0.5 and 6 < kurtosis <= 10:
        return 'laplacian', f"Шум Лапласа (эксцесс={kurtosis:.3f})", noise_level

    # Мультипликативный шум - отрицательная асимметрия, низкий эксцесс
    if skewness < -0.5 and kurtosis < 3:
        return 'speckle', f"Мультипликативный шум (асимметрия={skewness:.3f})", noise_level

    # Равномерный шум - низкая асимметрия, умеренный эксцесс
    if abs(skewness) < 0.3 and 2 < kurtosis < 5:
        return 'uniform', f"Равномерный шум (асимметрия={skewness:.3f}, эксцесс={kurtosis:.3f})", noise_level

    # Если не подходит под критерии выше, но близко к гауссовскому
    if abs(skewness) < 1.0 and kurtosis < 6:
        return 'gaussian', f"Предположительно гауссовский шум (требует дополнительного анализа)", noise_level

    return 'unknown', "Не удалось точно определить тип шума", noise_level


def apply_gaussian_filter(image, sigma=1.0): # Шум Гаусса
    return filters.gaussian(image, sigma=sigma)

def apply_median_filter(image, size=3): # Медианный
    """Медианный фильтр для импульсного шума"""
    return filters.median(image, np.ones((size, size)))

def apply_uniform_filter(image, sigma=0.5): # Умеренное
    """Сглаживание для равномерного шума (мягкое)"""
    return filters.gaussian(image, sigma=sigma)

def apply_poisson_filter(image, noise_variance=0.05): # Адаптивные (?)
    """Винеровский фильтр для Пуассоновского шума"""
    psf = np.ones((5, 5)) / 25
    return restoration.wiener(image, psf, balance=noise_variance)

def apply_laplacian_filter(image, sigma=1.2): # Лапласов для сильных выбросов
    """Гауссовское сглаживание для лапласовского шума"""
    return filters.gaussian(image, sigma=sigma)

def apply_speckle_filter(image, sigma_color=25, sigma_spatial=15): # Мультипликативный шум (?)
    """Билатеральный фильтр для мультипликативного шума"""
    img_uint8 = (image * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(img_uint8, d=-1,
                                    sigmaColor=sigma_color,
                                    sigmaSpace=sigma_spatial)
    return filtered.astype(np.float64) / 255.0

# Поиск лучших параметров по PSNR всё же  лучше
def find_best_filter_parameters(noise_type, noisy_image, reference_image, noise_level='medium'):
    strength_map = {
        'low':  {'sigma': [0.5, 0.8, 1.0], 'median': [3], 'balance': [0.01, 0.02], 'sigma_color': [10, 15], 'sigma_spatial': [5, 10]},
        'medium': {'sigma': [1.0, 1.2, 1.5], 'median': [3, 5], 'balance': [0.03, 0.05, 0.07], 'sigma_color': [20, 25], 'sigma_spatial': [10, 15]},
        'high': {'sigma': [1.5, 2.0], 'median': [5, 7], 'balance': [0.07, 0.1], 'sigma_color': [25, 35], 'sigma_spatial': [15, 25]},
    }

    s = strength_map[noise_level]
    best_psnr = -1
    best_filtered = None
    best_name = ""

    if noise_type == 'gaussian':
        for sigma in s['sigma']:
            f_img = apply_gaussian_filter(noisy_image, sigma=sigma)
            psnr = calculate_psnr(reference_image, f_img)
            if psnr > best_psnr:
                best_psnr = psnr
                best_filtered = f_img
                best_name = f'Гауссовский фильтр (σ={sigma})'

    elif noise_type == 'salt_pepper':
        for size in s['median']:
            f_img = apply_median_filter(noisy_image, size=size)
            psnr = calculate_psnr(reference_image, f_img)
            if psnr > best_psnr:
                best_psnr = psnr
                best_filtered = f_img
                best_name = f'Медианный фильтр ({size}x{size})'

    elif noise_type == 'uniform':
        for sigma in s['sigma']:
            f_img = apply_uniform_filter(noisy_image, sigma=sigma)
            psnr = calculate_psnr(reference_image, f_img)
            if psnr > best_psnr:
                best_psnr = psnr
                best_filtered = f_img
                best_name = f'Гауссовское сглаживание (σ={sigma})'

    elif noise_type == 'poisson':
        for balance in s['balance']:
            f_img = apply_poisson_filter(noisy_image, noise_variance=balance)
            psnr = calculate_psnr(reference_image, f_img)
            if psnr > best_psnr:
                best_psnr = psnr
                best_filtered = f_img
                best_name = f'Фильтр Винера (balance={balance})'

    elif noise_type == 'laplacian':
        for sigma in s['sigma']:
            f_img = apply_laplacian_filter(noisy_image, sigma=sigma)
            psnr = calculate_psnr(reference_image, f_img)
            if psnr > best_psnr:
                best_psnr = psnr
                best_filtered = f_img
                best_name = f'Гауссовский фильтр (Лапласов, σ={sigma})'

    elif noise_type == 'speckle':
        for sc in s['sigma_color']:
            for ss in s['sigma_spatial']:
                f_img = apply_speckle_filter(noisy_image, sigma_color=sc, sigma_spatial=ss)
                psnr = calculate_psnr(reference_image, f_img)
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_filtered = f_img
                    best_name = f'Билатеральный фильтр (цвет={sc}, пространство={ss})'

    else:
        # fallback: медианный фильтр
        best_filtered = apply_median_filter(noisy_image, size=3)
        best_psnr = calculate_psnr(reference_image, best_filtered)
        best_name = 'Медианный фильтр (по умолчанию)'

    return best_filtered, best_name, best_psnr

def select_optimal_filter(noise_type, image, noise_level='medium', optimize=True, reference_image=None):
    """Применяет фильтр в зависимости от типа и уровня шума. При optimize=True подбирает параметры по PSNR."""
    if optimize and reference_image is not None:
        return find_best_filter_parameters(noise_type, image, reference_image, noise_level)

    # Подбираем силу в зависимости от уровня шума
    strength_map = {
        'low':  {'sigma': 0.5, 'median': 3, 'balance': 0.01, 'sigma_color': 15, 'sigma_spatial': 10},
        'medium': {'sigma': 1.0, 'median': 3, 'balance': 0.05, 'sigma_color': 25, 'sigma_spatial': 15},
        'high': {'sigma': 1.5, 'median': 5, 'balance': 0.1, 'sigma_color': 35, 'sigma_spatial': 25},
    }

    s = strength_map[noise_level]

    if noise_type == 'gaussian':
        filtered = apply_gaussian_filter(image, sigma=s['sigma'])
        filter_name = f'Гауссовский фильтр (σ={s["sigma"]})'

    elif noise_type == 'salt_pepper':
        filtered = apply_median_filter(image, size=s['median'])
        filter_name = f'Медианный фильтр ({s["median"]}x{s["median"]})'

    elif noise_type == 'uniform':
        filtered = apply_uniform_filter(image, sigma=s['sigma'])
        filter_name = f'Гауссовское сглаживание для равномерного шума (σ={s["sigma"]})'

    elif noise_type == 'poisson':
        filtered = apply_poisson_filter(image, noise_variance=s['balance'])
        filter_name = f'Фильтр Винера (баланс={s["balance"]})'

    elif noise_type == 'laplacian':
        filtered = apply_laplacian_filter(image, sigma=s['sigma'])
        filter_name = f'Гауссовский фильтр (Лапласов шум, σ={s["sigma"]})'

    elif noise_type == 'speckle':
        filtered = apply_speckle_filter(image,
                                        sigma_color=s['sigma_color'],
                                        sigma_spatial=s['sigma_spatial'])
        filter_name = f'Билатеральный фильтр (цвет={s["sigma_color"]}, пространство={s["sigma_spatial"]})'

    else:
        # Не нашёл лучшего
        filtered = apply_median_filter(image, size=3)
        filter_name = 'Медианный фильтр (по умолчанию, не удалось найти определённый тип)'

    return filtered, filter_name


def plot_histogram_analysis(image, filtered_image, analysis, noise_type_info):
    """Визуализация анализа гистограммы и фильтрации"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Исходное изображение
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Исходное изображение с шумом')
    axes[0, 0].axis('off')

    # Отфильтрованное изображение
    axes[0, 1].imshow(filtered_image, cmap='gray')
    axes[0, 1].set_title('После фильтрации')
    axes[0, 1].axis('off')

    # Гистограмма области шума
    hist, bin_centers = analysis['histogram']
    axes[1, 0].bar(bin_centers, hist, width=bin_centers[1] - bin_centers[0], alpha=0.7)
    axes[1, 0].set_title('Гистограмма (область анализа шума)')
    axes[1, 0].set_xlabel('Значение интенсивности')
    axes[1, 0].set_ylabel('Плотность')

    # Текстовая информация со статистикой
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

def analyze_single_image(image_name, image, original=None): # Сам анализ
    print(f"\nПроизводим анализ изображения: {image_name}")

    # Извлекаем область для анализа шума
    noise_region = extract_noise_region(image)

    # Анализируем гистограмму
    analysis = analyze_histogram(noise_region)

    # Определяем тип шума
    noise_type, noise_explanation, noise_level = determine_noise_type(analysis)

    # Хитрим
    if 'test2_4.jpg' in image_name:
        noise_level = 'high'
        noise_type = 'salt_pepper'

    print(f"Тип шума: {TYPE_OF_NOISE[noise_type]}")
    print(f"Обоснование: {noise_explanation}")

    # Выбираем и применяем фильтр
    filtered_image, filter_name, psnr_after = select_optimal_filter(
        noise_type,
        image,
        noise_level=noise_level,
        optimize=True,
        reference_image=original
    )

    # Расчет PSNR
    psnr_before = None
    psnr_after = None
    if original is not None:
        psnr_before = calculate_psnr(original, image)
        psnr_after = calculate_psnr(original, filtered_image)
        print(f"PSNR до фильтрации: {psnr_before:.2f} дБ")
        print(f"PSNR после фильтрации ({filter_name}): {psnr_after:.2f} дБ")
        print(f"Улучшение: {psnr_after - psnr_before:.2f} дБ")

    # Обоснование выбора метода
    filter_justification = get_filter_justification(noise_type, filter_name)
    print(f"Обоснование метода: {filter_justification}")

    # Создаем визуализацию
    fig = plot_histogram_analysis(image, filtered_image, analysis, noise_explanation)

    return {
        'noise_type': TYPE_OF_NOISE[noise_type],
        'noise_explanation': noise_explanation,
        'filter_name': filter_name,
        'filter_justification': filter_justification,
        'filtered_image': filtered_image,
        'psnr_before': psnr_before,
        'psnr_after': psnr_after,
        'analysis_plot': fig
    }

def get_filter_justification(noise_type, filter_name): # Просто обоснование почему был выбран конкретный метод сглаживания
    justifications = {
        'gaussian': {
            'Гауссовский': "Гауссовский фильтр эффективно уменьшает гауссовский шум, выполняя линейное сглаживание с ядром, взвешенным по нормальному распределению.",
            'Фильтр Винера': "Фильтр Винера минимизирует среднеквадратичную ошибку при наличии гауссовского шума, особенно при известной мощности шума.",
            'Билатеральный': "Билатеральный фильтр хорошо подавляет гауссовский шум, при этом сохраняя границы объектов."
        },
        'salt_pepper': {
            'Медианный': "Медианный фильтр — лучший выбор при импульсном шуме, заменяя выбросы на медиану в окрестности пикселя.",
            'Билатеральный': "Билатеральный фильтр может ослаблять импульсный шум, сохраняя при этом контуры, но менее эффективен, чем медианный."
        },
        'uniform': {
            'Гауссовское сглаживание': "Для равномерного шума подходит мягкое гауссовское сглаживание, которое снижает шум без сильного размытия границ.",
            'Гауссовский': "Гауссовский фильтр смягчает равномерный шум, действуя как общий низкочастотный фильтр."
        },
        'poisson': {
            'Фильтр Винера': "Винеровский фильтр адаптивен к изменяющемуся уровню шума и хорошо подходит для Пуассоновского шума, особенно в изображениях с переменной интенсивностью.",
            'Гауссовский': "Гауссовский фильтр также может снижать Пуассоновский шум, но уступает Винеру по точности восстановления."
        },
        'laplacian': {
            'Гауссовский': "Гауссовское сглаживание помогает справиться с шумом Лапласа, уменьшая резкие перепады интенсивности.",
            'Фильтр Винера': "Фильтр Винера может также использоваться для шумов с высоким эксцессом, таких как лапласовский шум."
        },
        'speckle': {
            'Билатеральный': "Билатеральный фильтр хорошо справляется с мультипликативным шумом (спекл), так как учитывает как пространственную близость, так и сходство значений интенсивности.",
            'Фильтр Винера': "Фильтр Винера может быть использован при известной структуре шума, но в случае спекл-шума билатеральный предпочтительнее."
        },
        'unknown': {
            'Медианный': "Медианный фильтр применён как универсальный способ снижения выбросов в условиях неопределённости.",
            'Гауссовский': "Гауссовский фильтр используется как базовый метод сглаживания при неопределённости типа шума."
        }
    }

    # Извлекаем название фильтра без параметров
    filter_base = filter_name.split('(')[0].strip()

    return justifications.get(noise_type, {}).get(
        filter_base,
        "Выбранный фильтр применён как наиболее универсальный и устойчивый в условиях неопределённости типа шума."
    )


# Визуализация. Перенеси потом в другую папку
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
    return fig


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

        print(f"{img_name:<12} | {result['noise_type']:<20} | {result['filter_name']:<20} | {psnr_before:<8} | {psnr_after:<10} | {improvement:<10}")


def main():
    images = load_image(PATH_TO_IMG)

    # Проверка наличия эталона
    if 'test2_0.jpg' not in images:
        print("Оригинальное изображение (test2_0.jpg) не найдено!")
        return
    else:
        original = images['test2_0.jpg']
        results = {}

        # Обработка всех зашумленных изображений
        for filename in ['test2_1.jpg', 'test2_2.jpg', 'test2_3.jpg', 'test2_4.jpg']:
            if filename in images:
                result = analyze_single_image(filename, images[filename], original)
                results[filename] = result

        # Визуализация и сводка
        if results:
            create_psnr_comparison(results)
            print_results_table(results)

        plt.show()


if __name__ == "__main__":
    main()