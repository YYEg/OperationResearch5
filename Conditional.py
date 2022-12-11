import numpy as np
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def Conditional_F(x):
    return (x[0] ** 2) + (5 * x[1] ** 2) - (2 * x[0] * x[1]) - (2 * x[0]) #целевая функция

def ff_der1(x):
    der1 = np.zeros_like(x)
    der1[0] = 2*x[0] - 2*x[1] - 2
    der1[1] = 10*x[1] - 2*x[0]
    return der1

ineq_cons = {'type': 'ineq',
             'fun': lambda x: np.array([20 - x[0] - 4 * x[1], -4 + x[0] + x[1]]),
             'jac': lambda x: np.array([[-1.0, -4.0], [1, 1]])
            }
bnd = [(0,float("inf")),
        (1, 12)]
x0 = np.array([2, 2])


opt = minimize(Conditional_F, x0, method='SLSQP', jac = ff_der1,
               constraints=[ineq_cons],
               options={'ftol': 1e-9, 'disp': True},
               bounds=bnd)
print(opt)

fig = plt.figure(figsize=[15, 10])
ax = fig.add_subplot(projection='3d')

# Задаем угол обзора
ax.view_init(45, 30)

# определяем область отрисовки графика
X = np.arange(0, 2, 0.1)
Y = np.arange(0, 4, 0.1)
X, Y = np.meshgrid(X, Y)
Z = Conditional_F(np.array([X, Y]))

# Рисуем поверхность
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
plt.title(r'Условная оптимизация', fontsize=16, y=1.05)

plt.show()
