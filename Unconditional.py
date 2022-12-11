import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm

x0 = np.array([1, 1])

def Unconditional_F(x):
    return (x[0] ** 2) + (5 * x[1] ** 2) - (2 * x[0] * x[1]) - (2 * x[0]) #целевая функция
def ff_der(x):
    der = np.zeros_like(x)
    der[0] = 2 * x[0] - 2 * x[1] - 2
    der[1] = 10 * x[1] - 2 * x[0]
    return der

opt = minimize(Unconditional_F, x0, method='BFGS', jac=ff_der, options={'disp': True})
print(opt)

fig = plt.figure(figsize=[15, 10])
ax = fig.add_subplot(projection='3d')

# Задаем угол обзора
ax.view_init(45, 30)

# определяем область отрисовки графика
X = np.arange(-20, 20, 1)
Y = np.arange(-20, 20, 1)
X, Y = np.meshgrid(X, Y)
Z = Unconditional_F(np.array([X, Y]))

# Рисуем поверхность
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
plt.title(r'Безусловная оптимизация', fontsize=16, y=1.05)
plt.show()
