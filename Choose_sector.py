import cv2

# Исходное изображение
image = cv2.imread("Data/test2_00.jpg")

# Координаты: верхний левый угол (x1, y1) и нижний правый угол (x2, y2)
start_point = (50, 100)
end_point = (150, 200)

# Цвет прямоугольника (BGR) и толщина линии
color = (0, 255, 0)  # зелёный
thickness = 2

# Нарисовать
cv2.rectangle(image, start_point, end_point, color, thickness)

# Показать
cv2.imshow("Rect", image)
cv2.waitKey(0)
cv2.destroyAllWindows()