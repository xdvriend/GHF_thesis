import math as m

def calculate_coordinates(number_of_angles, circumscribed_radius):
    r = circumscribed_radius
    x_values = []
    y_values = []
    for n in range(number_of_angles):
        theta = n * (360 / number_of_angles)
        x_values.append(r * m.sin(m.radians(theta)))
        y_values.append(r * m.cos(m.radians(theta)))
        n +=1
    return x_values, y_values

print(calculate_coordinates(3, 0.57735))
print(calculate_coordinates(4, 0.707107))
print(calculate_coordinates(5, 0.850651))
print(calculate_coordinates(7, 1.15238))
print(calculate_coordinates(8, 1.30656))
print(calculate_coordinates(9, 1.4619))
print(calculate_coordinates(10, 1.61803))
print(calculate_coordinates(11, 1.77473))
print(calculate_coordinates(12, 1.93185))
print(calculate_coordinates(13, 2.08929))
print(calculate_coordinates(14, 2.24698))
print(calculate_coordinates(15, 2.40487))










