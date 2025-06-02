import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage import filters, restoration
from skimage.metrics import peak_signal_noise_ratio
from itertools import combinations

# Все изображения
ALL_IMAGES = {
    'test2_1.jpg': 'Data/test2_1.jpg',
    'test2_2.jpg': 'Data/test2_2.jpg',
    'test2_3.jpg': 'Data/test2_3.jpg',
    'test2_4.jpg': 'Data/test2_4.jpg'
}


# Маппинг фильтров к типам шума для обратного определения
FILTER_TO_NOISE = {
    'gaussian': ['gaussian', 'uniform'],
    'median': ['salt_pepper'],
    'wiener': ['poisson', 'gaussian'],
    'bilateral': ['speckle', 'gaussian']
}

# Обратный маппинг - какие шумы лучше всего убираются конкретными фильтрами
NOISE_SIGNATURES = {
    'gaussian': {'primary_filters': ['gaussian'], 'secondary_filters': ['wiener', 'bilateral']},
    'salt_pepper': {'primary_filters': ['median'], 'secondary_filters': []},
    'uniform': {'primary_filters': ['gaussian'], 'secondary_filters': []},
    'poisson': {'primary_filters': ['wiener'], 'secondary_filters': ['gaussian']},
    'speckle': {'primary_filters': ['bilateral'], 'secondary_filters': ['wiener']},
    'laplacian': {'primary_filters': ['gaussian'], 'secondary_filters': ['wiener']}
}


def apply_filters(image):
    """Применяет все доступные фильтры с разными параметрами"""
    filters_results = {}

    # Гауссовский фильтр
    for sigma in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        filtered = filters.gaussian(image, sigma=sigma)
        filters_results[f'gaussian_s{sigma}'] = {
            'image': filtered,
            'filter_type': 'gaussian',
            'params': {'sigma': sigma},
            'name': f'Гауссовский (σ={sigma})'
        }

    # Медианный фильтр
    for size in [3, 5, 7]:
        filtered = filters.median(image, np.ones((size, size)))
        filters_results[f'median_s{size}'] = {
            'image': filtered,
            'filter_type': 'median',
            'params': {'size': size},
            'name': f'Медианный ({size}x{size})'
        }

    # Фильтр Винера
    for balance in [0.01, 0.02, 0.05, 0.07, 0.1]:
        psf = np.ones((5, 5)) / 25
        filtered = restoration.wiener(image, psf, balance=balance)
        filters_results[f'wiener_b{balance}'] = {
            'image': filtered,
            'filter_type': 'wiener',
            'params': {'balance': balance},
            'name': f'Винера (balance={balance})'
        }

    # Билатеральный фильтр
    for sigma_color in [10, 15, 20, 25, 35]:
        for sigma_spatial in [5, 10, 15, 25]:
            img_uint8 = (image * 255).astype(np.uint8)
            filtered = cv2.bilateralFilter(img_uint8, d=-1,
                                           sigmaColor=sigma_color,
                                           sigmaSpace=sigma_spatial)
            filtered = filtered.astype(np.float64) / 255.0

            key = f'bilateral_c{sigma_color}_s{sigma_spatial}'
            filters_results[key] = {
                'image': filtered,
                'filter_type': 'bilateral',
                'params': {'sigma_color': sigma_color, 'sigma_spatial': sigma_spatial},
                'name': f'Билатеральный (c={sigma_color}, s={sigma_spatial})'
            }

    return filters_results


def find_best_cascade_filters(noisy_image, original_image, max_depth=3):
    """
    Находит оптимальную последовательность фильтров (каскад) глубиной до max_depth
    """
    current_image = noisy_image.copy()
    original_psnr = peak_signal_noise_ratio(original_image, current_image, data_range=1.0)

    cascade_results = []
    best_cascade = []
    best_psnr = original_psnr

    print(f"Исходный PSNR: {original_psnr:.2f} дБ")

    for depth in range(1, max_depth + 1):
        print(f"\nПоиск на глубине {depth}...")

        # Применяем все фильтры к текущему изображению
        all_filters = apply_filters(current_image)

        # Находим лучший фильтр для текущего шага
        step_best_psnr = -1
        step_best_filter = None
        step_best_key = None

        for filter_key, filter_data in all_filters.items():
            psnr = peak_signal_noise_ratio(original_image, filter_data['image'], data_range=1.0)
            if psnr > step_best_psnr:
                step_best_psnr = psnr
                step_best_filter = filter_data
                step_best_key = filter_key

        # Проверяем улучшение
        improvement = step_best_psnr - best_psnr
        print(f"Лучший фильтр: {step_best_filter['name']}")
        print(f"PSNR: {step_best_psnr:.2f} дБ (улучшение: {improvement:.2f} дБ)")

        # Если улучшение значительное, добавляем в каскад
        if improvement > 0.1:  # порог улучшения
            best_cascade.append({
                'filter': step_best_filter,
                'psnr': step_best_psnr,
                'improvement': improvement
            })
            current_image = step_best_filter['image'].copy()
            best_psnr = step_best_psnr
        else:
            print("Улучшение незначительное, останавливаем каскад")
            break

    return best_cascade, best_psnr


def analyze_noise_from_cascade(cascade_results):
    """
    Определяет тип шума на основе последовательности лучших фильтров
    """
    if not cascade_results:
        return "unknown", "Не удалось применить ни одного фильтра"

    # Собираем типы использованных фильтров
    used_filters = [step['filter']['filter_type'] for step in cascade_results]
    filter_sequence = " -> ".join([step['filter']['name'] for step in cascade_results])

    # Анализируем паттерны
    noise_candidates = []

    # 1. Анализ по доминирующему фильтру (первый и самый эффективный)
    primary_filter = used_filters[0]

    # 2. Анализ по комбинации фильтров
    filter_set = set(used_filters)

    # Логика определения шума
    if primary_filter == 'median':
        if len(used_filters) == 1:
            noise_candidates.append(('salt_pepper', 'Импульсный шум - эффективен только медианный фильтр'))
        else:
            noise_candidates.append(('mixed', 'Смешанный шум с импульсной компонентой'))

    elif primary_filter == 'gaussian':
        if 'wiener' in filter_set:
            noise_candidates.append(('gaussian', 'Гауссовский шум - эффективны гауссовский и винеровский фильтры'))
        elif 'bilateral' in filter_set:
            noise_candidates.append(('mixed', 'Смешанный шум с гауссовской компонентой'))
        else:
            noise_candidates.append(('gaussian', 'Гауссовский/равномерный шум'))

    elif primary_filter == 'wiener':
        if 'gaussian' in filter_set:
            noise_candidates.append(('poisson', 'Пуассоновский шум - винеровский фильтр с гауссовским'))
        else:
            noise_candidates.append(('poisson', 'Пуассоновский шум'))

    elif primary_filter == 'bilateral':
        if len(used_filters) == 1:
            noise_candidates.append(('speckle', 'Мультипликативный (спекл) шум'))
        else:
            noise_candidates.append(('mixed', 'Смешанный шум со спекл-компонентой'))

    # Если несколько разных типов фильтров - вероятно смешанный шум
    if len(filter_set) >= 3:
        noise_candidates.append(('mixed', 'Сложный смешанный шум - требует множественной фильтрации'))

    # Выбираем наиболее вероятный тип шума
    if noise_candidates:
        primary_noise = noise_candidates[0]
        return primary_noise[0], f"{primary_noise[1]}. Последовательность фильтров: {filter_sequence}"
    else:
        return "unknown", f"Неопределенный тип шума. Последовательность фильтров: {filter_sequence}"


def statistical_noise_analysis(image):
    """
    Дополнительный статистический анализ для подтверждения типа шума
    """
    # Выделяем однородную область для анализа
    h, w = image.shape
    region = image[h // 4:3 * h // 4, w // 4:3 * w // 4]  # центральная область

    # Вычисляем статистики
    mean_val = np.mean(region)
    std_val = np.std(region)
    skewness = stats.skew(region.flatten())
    kurtosis = stats.kurtosis(region.flatten())

    # Тест на нормальность
    if len(region.flatten()) > 5000:
        sample = np.random.choice(region.flatten(), 5000, replace=False)
    else:
        sample = region.flatten()

    shapiro_stat, shapiro_p = stats.shapiro(sample)

    analysis_hints = []

    # Подсказки по статистикам
    if shapiro_p > 0.05 and abs(skewness) < 0.5 and abs(kurtosis) < 3:
        analysis_hints.append("Статистика указывает на гауссовский шум")
    elif kurtosis > 10:
        analysis_hints.append("Высокий эксцесс указывает на импульсный шум")
    elif abs(mean_val - std_val ** 2) < 0.1 * mean_val and mean_val > 0.1:
        analysis_hints.append("Соотношение среднего и дисперсии указывает на пуассоновский шум")
    elif skewness < -0.5:
        analysis_hints.append("Отрицательная асимметрия может указывать на мультипликативный шум")

    return {
        'mean': mean_val,
        'std': std_val,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'shapiro_p': shapiro_p,
        'hints': analysis_hints
    }


def comprehensive_noise_analysis(noisy_image, original_image, max_depth=3):
    """
    Комплексный анализ изображения с определением типа шума
    """
    print("=== КОМПЛЕКСНЫЙ АНАЛИЗ ШУМА ===\n")

    # 1. Находим оптимальный каскад фильтров
    cascade_results, final_psnr = find_best_cascade_filters(noisy_image, original_image, max_depth)

    # 2. Определяем тип шума по каскаду
    noise_type, cascade_explanation = analyze_noise_from_cascade(cascade_results)

    # 3. Дополнительный статистический анализ
    stats_analysis = statistical_noise_analysis(noisy_image)

    # 4. Объединяем результаты
    print(f"\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===")
    print(f"Определенный тип шума: {noise_type}")
    print(f"Обоснование: {cascade_explanation}")

    print(f"\nСтатистические подсказки:")
    for hint in stats_analysis['hints']:
        print(f"  • {hint}")

    original_psnr = peak_signal_noise_ratio(original_image, noisy_image, data_range=1.0)
    print(f"\nИтоговые метрики:")
    print(f"  • Исходный PSNR: {original_psnr:.2f} дБ")
    print(f"  • Финальный PSNR: {final_psnr:.2f} дБ")
    print(f"  • Общее улучшение: {final_psnr - original_psnr:.2f} дБ")

    return {
        'noise_type': noise_type,
        'explanation': cascade_explanation,
        'cascade': cascade_results,
        'statistics': stats_analysis,
        'psnr_original': original_psnr,
        'psnr_final': final_psnr,
        'improvement': final_psnr - original_psnr
    }

def all_photo_analyz(orig_img, IMAGES):
    for i in range(len(IMAGES)):
        original = orig_img
        noisy = IMAGES[i]

        result = comprehensive_noise_analysis(noisy, original, max_depth=3)


if __name__ == "__main__":
    original = cv2.IMREAD_GRAYSCALE("Data/test2_0.jpg")
    all_photo_analyz(original, I)