import os
from pathlib import Path
import cv2
import numpy as np
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
    'mixed': 'Смешанный шум',
    'unknown': 'Неопределённый шум'
}

PATH_TO_IMG = {
    'test2_0.jpg': r'Data/test2_0.jpg',
    'test2_1.jpg': r'Data/test2_1.jpg',
    'test2_2.jpg': r'Data/test2_2.jpg',
    'test2_3.jpg': r'Data/test2_3.jpg',
    'test2_4.jpg': r'Data/test2_4.jpg'
}


def load_image(paths):
    images = {}
    for name, path in paths.items():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images[name] = img.astype(np.float64) / 255.0
        else:
            print(f"Предупреждение: не удалось загрузить изображение {path}")
    return images


def calculate_psnr(original, noisy):
    return peak_signal_noise_ratio(original, noisy, data_range=1.0)


def extract_noise_region(image):
    """Извлечение всей области изображения для анализа шума"""
    h, w = image.shape
    return image[0:h, 0:w]


def apply_gaussian_filter(image, sigma=1.0):
    return filters.gaussian(image, sigma=sigma)


def apply_median_filter(image, size=3):
    return filters.median(image, np.ones((size, size)))


def apply_poisson_filter(image, noise_variance=0.05):
    psf = np.ones((5, 5)) / 25
    return restoration.wiener(image, psf, balance=noise_variance)


def apply_speckle_filter(image, sigma_color=25, sigma_spatial=15):
    img_uint8 = (image * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(img_uint8, d=-1,
                                   sigmaColor=sigma_color,
                                   sigmaSpace=sigma_spatial)
    return filtered.astype(np.float64) / 255.0


def find_best_cascade_filters(noisy_image, original_image, max_depth=4):
    """
    Находит оптимальную последовательность фильтров (каскад) глубиной до max_depth
    """
    current_image = noisy_image.copy()
    original_psnr = calculate_psnr(original_image, current_image)

    cascade_results = []
    best_psnr = original_psnr

    # Параметры для перебора
    params = {
        'gaussian': {'sigma': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]},
        'median': {'size': [3, 5, 7]},
        'wiener': {'balance': [0.01, 0.02, 0.05, 0.07, 0.1]},
        'bilateral': {
            'sigma_color': [10, 15, 20, 25, 35],
            'sigma_spatial': [5, 10, 15, 25]
        }
    }

    for depth in range(1, max_depth + 1):
        step_best_psnr = -1
        step_best_result = None

        # Гауссовский фильтр
        for sigma in params['gaussian']['sigma']:
            filtered = apply_gaussian_filter(current_image, sigma=sigma)
            psnr = calculate_psnr(original_image, filtered)
            if psnr > step_best_psnr:
                step_best_psnr = psnr
                step_best_result = {
                    'filter': 'gaussian',
                    'params': {'sigma': sigma},
                    'filtered_img': filtered,
                    'psnr': psnr,
                    'filter_name': f'Гауссовский фильтр (σ={sigma})'
                }

        # Медианный фильтр
        for size in params['median']['size']:
            filtered = apply_median_filter(current_image, size=size)
            psnr = calculate_psnr(original_image, filtered)
            if psnr > step_best_psnr:
                step_best_psnr = psnr
                step_best_result = {
                    'filter': 'median',
                    'params': {'size': size},
                    'filtered_img': filtered,
                    'psnr': psnr,
                    'filter_name': f'Медианный фильтр ({size}x{size})'
                }

        # Фильтр Винера
        for balance in params['wiener']['balance']:
            filtered = apply_poisson_filter(current_image, noise_variance=balance)
            psnr = calculate_psnr(original_image, filtered)
            if psnr > step_best_psnr:
                step_best_psnr = psnr
                step_best_result = {
                    'filter': 'wiener',
                    'params': {'balance': balance},
                    'filtered_img': filtered,
                    'psnr': psnr,
                    'filter_name': f'Фильтр Винера (balance={balance})'
                }

        # Билатеральный фильтр
        for sigma_color in params['bilateral']['sigma_color']:
            for sigma_spatial in params['bilateral']['sigma_spatial']:
                filtered = apply_speckle_filter(
                    current_image,
                    sigma_color=sigma_color,
                    sigma_spatial=sigma_spatial
                )
                psnr = calculate_psnr(original_image, filtered)
                if psnr > step_best_psnr:
                    step_best_psnr = psnr
                    step_best_result = {
                        'filter': 'bilateral',
                        'params': {
                            'sigma_color': sigma_color,
                            'sigma_spatial': sigma_spatial
                        },
                        'filtered_img': filtered,
                        'psnr': psnr,
                        'filter_name': f'Билатеральный фильтр (цвет={sigma_color}, пространство={sigma_spatial})'
                    }

        # Проверяем улучшение
        improvement = step_best_psnr - best_psnr

        if improvement > 0:  # порог улучшения
            cascade_results.append({
                'step': depth,
                'result': step_best_result,
                'improvement': improvement
            })
            current_image = step_best_result['filtered_img'].copy()
            best_psnr = step_best_psnr
        else:
            break  # Если улучшение незначительное, останавливаем каскад

    return cascade_results, best_psnr


def determine_noise_type_from_cascade(cascade_results):
    """Определяет тип шума на основе последовательности лучших фильтров"""
    if not cascade_results:
        return 'unknown', "Не удалось применить ни одного фильтра эффективно"

    # Собираем типы использованных фильтров
    used_filters = [step['result']['filter'] for step in cascade_results]
    filter_sequence = " -> ".join([step['result']['filter_name'] for step in cascade_results])

    # Анализируем первый (самый эффективный) фильтр
    primary_filter = used_filters[0]

    # Определяем тип шума
    if primary_filter == 'median':
        if len(used_filters) == 1:
            return 'salt_pepper', f"Импульсный шум - наиболее эффективен медианный фильтр. Последовательность: {filter_sequence}"
        else:
            return 'mixed', f"Смешанный шум с импульсной компонентой. Последовательность: {filter_sequence}"

    elif primary_filter == 'gaussian':
        if 'wiener' in used_filters:
            return 'gaussian', f"Гауссовский шум - эффективны гауссовский и винеровский фильтры. Последовательность: {filter_sequence}"
        elif len(used_filters) > 1:
            return 'mixed', f"Смешанный шум с гауссовской компонентой. Последовательность: {filter_sequence}"
        else:
            return 'gaussian', f"Гауссовский/равномерный шум. Последовательность: {filter_sequence}"

    elif primary_filter == 'wiener':
        if 'gaussian' in used_filters:
            return 'poisson', f"Пуассоновский шум - винеровский фильтр с гауссовским сглаживанием. Последовательность: {filter_sequence}"
        else:
            return 'poisson', f"Пуассоновский шум. Последовательность: {filter_sequence}"

    elif primary_filter == 'bilateral':
        if len(used_filters) == 1:
            return 'speckle', f"Мультипликативный (спекл) шум. Последовательность: {filter_sequence}"
        else:
            return 'mixed', f"Смешанный шум со спекл-компонентой. Последовательность: {filter_sequence}"

    # Если несколько разных типов фильтров - вероятно сложный смешанный шум
    if len(set(used_filters)) >= 3:
        return 'mixed', f"Сложный смешанный шум - требует множественной фильтрации. Последовательность: {filter_sequence}"

    return 'unknown', f"Неопределенный тип шума. Последовательность: {filter_sequence}"

def comprehensive_analysis_single_image(img_name, noisy_image, original_image, max_depth=4):
    """Комплексный анализ одного изображения с каскадной фильтрацией"""
    print(f"АНАЛИЗ ИЗОБРАЖЕНИЯ: {img_name}")

    original_psnr = calculate_psnr(original_image, noisy_image)
    print(f"Исходный PSNR: {original_psnr:.2f} дБ")

    # 1. Находим оптимальный каскад фильтров
    cascade_results, final_psnr = find_best_cascade_filters(noisy_image, original_image, max_depth)

    # 3. Дополнительный статистический анализ области
    noise_region = extract_noise_region(noisy_image)
    mean_val = np.mean(noise_region)
    std_val = np.std(noise_region)
    skew = stats.skew(noise_region.flatten())
    kurt = stats.kurtosis(noise_region.flatten())
    p_value = stats.shapiro(noise_region.flatten()[:5000])[1]

    print(f"\nКаскад фильтрации:")
    total_improvement = 0
    for i, step in enumerate(cascade_results, 1):
        result = step['result']
        improvement = step['improvement']
        total_improvement += improvement
        print(f"  Шаг {i}: {result['filter_name']} (PSNR: {result['psnr']:.2f} дБ, +{improvement:.2f} дБ)")

    print(f"\nИтоговые метрики:")
    print(f"  • Финальный PSNR: {final_psnr:.2f} дБ")
    print(f"  • Общее улучшение: {total_improvement:.2f} дБ")

    # Возвращаем финальное отфильтрованное изображение
    final_filtered = cascade_results[-1]['result']['filtered_img'] if cascade_results else noisy_image

    # Сохраняем финальное изображение после каскада
    save_filtered_image(final_filtered, f"{img_name}")

    return {
        'cascade': cascade_results,
        'statistics': {
            'mean': mean_val,
            'std': std_val,
            'skewness': skew,
            'kurtosis': kurt,
            'shapiro_p': p_value
        },
        'psnr_original': original_psnr,
        'psnr_final': final_psnr,
        'improvement': total_improvement,
        'final_filtered_image': final_filtered
    }


def main_cascade_analysis():
    """Основная функция для анализа всех изображений с каскадной фильтрацией"""
    images = load_image(PATH_TO_IMG)

    if 'test2_0.jpg' not in images:
        print("Оригинальное изображение (test2_0.jpg) не найдено!")
        return

    original = images['test2_0.jpg']
    results = {}

    # Анализируем все зашумленные изображения
    for filename in ['test2_1.jpg', 'test2_2.jpg', 'test2_3.jpg', 'test2_4.jpg']:
        if filename in images:
            result = comprehensive_analysis_single_image(
                filename,
                images[filename],
                original,
                max_depth=4
            )
            results[filename] = result

    # Сводная таблица
    print(f"\n{'=' * 100}")
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print(f"{'=' * 100}")

    headers = ["Изображение", "Тип шума", "Количество фильтров", "PSNR до", "PSNR после", "Улучшение"]
    print(
        f"{headers[0]:<15} | {headers[1]:<20} | {headers[2]:<18} | {headers[3]:<8} | {headers[4]:<10} | {headers[5]:<10}")
    print("-" * 100)

    for img_name, result in results.items():
        noise_type = TYPE_OF_NOISE.get(result['noise_type'], result['noise_type'])
        num_filters = len(result['cascade'])
        psnr_before = f"{result['psnr_original']:.2f}"
        psnr_after = f"{result['psnr_final']:.2f}"
        improvement = f"{result['improvement']:.2f}"

        print(
            f"{img_name:<15} | {noise_type:<20} | {num_filters:<18} | {psnr_before:<8} | {psnr_after:<10} | {improvement:<10}")

    return results


def save_filtered_image(image, filename, folder="Data/better"):
    """Сохраняет изображение в указанную папку"""
    Path(folder).mkdir(parents=True, exist_ok=True)  # создать папку, если нет
    save_path = os.path.join(folder, filename)
    img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(save_path, img_uint8)

if __name__ == "__main__":
    main_cascade_analysis()