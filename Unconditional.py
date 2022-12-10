import numpy as np
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

x0 = np.array([1, 1])


def Unconditional_F(x):
    return (-x[0] ** 2) - (5 * x[1] ** 2) + (2 * x[0] * x[1]) + (2 * x[0]) #целевая функция


opt = minimize(Unconditional_F, x0, method='nelder-mead')
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
plt.show()
