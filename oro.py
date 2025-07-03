#este código calcula el valor de la eficiencia de scattering y extinción para un núcleo aquiral (sin corteza quiral)
#en función del radio de la esfera y de la longitud de onda del haz. Para este caso se obtuvieron valores del índice de refracción del oro,
# que depende de la longitud de onda del haz, y se interpolaron los datos disponibles para tener una mayor cantidad de los mismos. 
# Como estos valores de índice de refracción son complejos, existe absorción. Se representa en un colormap.
#La helicidad se cambia manualmente buscando sigma en el buscador. Recordar cambiarlo también en los títulos de las gráficas.

import numpy as np
from scipy.special import spherical_jn as besselj, spherical_yn as bessely
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

# Constantes físicas
eh = 2.5
c = 3e8  # velocidad de la luz en m/s
nm = 1e-9
fs = 1e-15
ps = 1e-12
THz = 1e12

# Parámetros del sistema
Delta_Omega_a = 175 * THz
tau_10 = tau_32 = 100 * fs
tau_21 = 50 * ps
Gamma_pump = 1e11
lambda_a = 777 * nm
# Línea NUEVA (después de imports)
def n_complex(lambda_nm):  # lambda_nm en nanómetros
    return f1(lambda_nm) + 1j * f2(lambda_nm)
n = np.array([1.49, 1.53, 1.53, 1.54, 1.48, 1.48, 1.5, 1.48, 1.46, 1.47, 1.46, 1.45, 1.38, 1.31, 1.04, 0.62, 0.43, 0.29, 0.21, 0.14, 0.13, 0.14, 0.16, 0.17, 0.22, 0.27])
k = np.array([1.878, 1.889, 1.893, 1.898, 1.883, 1.871, 1.886, 1.895, 1.933, 1.952, 1.958, 1.948, 1.914, 1.849, 1.833, 2.081, 2.455, 2.863, 3.272, 3.697, 4.103, 4.542, 5.083, 5.663, 6.35, 7.15])
lambda_um = np.array([0.2924, 0.3009, 0.3107, 0.3204, 0.3315, 0.3425, 0.3542, 0.3679, 0.3815, 0.3974, 0.4133, 0.4305, 0.4509, 0.4714, 0.4959, 0.5209, 0.5486, 0.5821, 0.6168, 0.65950, 0.7045, 0.756, 0.8211, 0.892, 0.984, 1.088])
lambda_nm = lambda_um * 1000
# Interpolación cúbica para ambas
f1 = interp1d(lambda_nm, n, kind='cubic')
f2 = interp1d(lambda_nm, k, kind='cubic')

def n_complex(lambda_nm_value):
    return f1(lambda_nm_value) + 1j * f2(lambda_nm_value)
# Constantes electromagnéticas
epsilon_0 = 8.85e-12
mu_0 = 4 * np.pi * 1e-7  # H/m
eta = 0.48

# Cálculo de sigma_b
sigma_b = (6 * np.pi * epsilon_0 * c**3 * eta) / (
    tau_21 * (2 * np.pi * c / lambda_a)**2 * np.sqrt(eh)
)

# Polarización circular (helicidad)
sigma = 1

# Permisividades y permeabilidades
epsilon_1 = 1.332**2
mu_1 = 1
mu_c = 1
n_1 = np.sqrt(epsilon_1)

# Redefinición del shell como núcleo: esfera homogénea

def n_sL(kappa, n_core):
    return n_core + kappa

def n_sR(kappa, n_core):
    return n_core - kappa

def m_2(kappa, n_core):
    return 0.5 * (n_sL(kappa, n_core) + n_sR(kappa, n_core))

def K_1(lambda_):
    return n_1 * (2 * np.pi / lambda_)

def k_0(lambda_):
    return n_1 * (2 * np.pi / lambda_)

def k_LL(lambda_, kappa, n_core):
    return n_sL(kappa, n_core) * (2 * np.pi / lambda_)

def k_R(lambda_, kappa, n_core):
    return n_sR(kappa, n_core) * (2 * np.pi / lambda_)

def k_II(lambda_, kappa, n_core):
    return 0.5 * (k_LL(lambda_, kappa, n_core) + k_R(lambda_, kappa, n_core))

def k_I_manual(lambda_, n_core):
    return n_core * 2 * np.pi / lambda_

def q_1(lambda_, r):
    return K_1(lambda_) * r

def x_0(lambda_, a):
    return n_1 * k_0(lambda_) * a

def v(lambda_, b):
    return k_0(lambda_) * b

def alpha(lambda_, a):
    return k_0(lambda_) * a

def NL(lambda_, kappa, n_core):
    return k_LL(lambda_, kappa, n_core) / k_0(lambda_)

def NR(lambda_, kappa, n_core):
    return k_R(lambda_, kappa, n_core) / k_0(lambda_)

def NII(lambda_, kappa, n_core):
    return k_II(lambda_, kappa, n_core) / k_0(lambda_)

def NI_manual(lambda_, n_core):
    return n_core / n_1


# -------------------- Funciones de Bessel esféricas --------------------

def z1(j, rho):
    return np.sqrt(np.pi / (2 * rho)) * besselj(j + 0.5, rho)

def eta1(j, rho):
    term1 = 0.5 * np.sqrt(np.pi / 2) * np.sqrt(1 / rho) * besselj(0.5 + j, rho)
    term2 = (
        np.sqrt(np.pi / 2)
        * (besselj(j - 0.5, rho) - besselj(j + 1.5, rho))
        / (2 * np.sqrt(1 / rho))
    )
    return (term1 + term2) / rho

def z2(j, rho):
    return np.sqrt(np.pi / (2 * rho)) * bessely(j + 0.5, rho)

def eta2(j, rho):
    term1 = 0.5 * np.sqrt(np.pi / 2) * np.sqrt(1 / rho) * bessely(0.5 + j, rho)
    term2 = (
        np.sqrt(np.pi / 2)
        * (bessely(j - 0.5, rho) - bessely(j + 1.5, rho))
        / (2 * np.sqrt(1 / rho))
    )
    return (term1 + term2) / rho

def z3(j, rho):
    return z1(j, rho) + 1j * z2(j, rho)

def eta3(j, rho):
    root_term = np.sqrt(np.pi / 2) * np.sqrt(1 / rho)
    j_term = besselj(0.5 + j, rho)
    y_term = bessely(0.5 + j, rho)

    deriv_j = (
        -0.5 * root_term * (1 / rho)**(1.0) * j_term
        + 0.5 * root_term * (besselj(j - 0.5, rho) - besselj(j + 1.5, rho))
    )

    deriv_y = (
        -0.5j * root_term * (1 / rho)**(1.0) * y_term
        + 0.5j * root_term * (bessely(j - 0.5, rho) - bessely(j + 1.5, rho))
    )

    return (root_term * (j_term + 1j * y_term) + rho * (deriv_j + deriv_y)) / rho

# -------------------- Funciones derivadas para coeficientes --------------------

def FNR(j, lambda_, a, kappa, n_core):
    rho1 = NR(lambda_, kappa, n_core) * alpha(lambda_, a)
    rho2 = NI_manual(lambda_, n_core) * alpha(lambda_, a)
    return NII(lambda_, kappa, n_core) * z2(j, rho1) * eta1(j, rho2) - NI_manual(lambda_, n_core) * eta2(j, rho1) * z1(j, rho2)

def GNR(j, lambda_, a, kappa, n_core):
    rho1 = NR(lambda_, kappa, n_core) * alpha(lambda_, a)
    rho2 = NI_manual(lambda_, n_core) * alpha(lambda_, a)
    return NI_manual(lambda_, n_core) * z2(j, rho1) * eta1(j, rho2) - NII(lambda_, kappa, n_core) * eta2(j, rho1) * z1(j, rho2)

def HNR(j, lambda_, a, kappa, n_core):
    rho1 = NR(lambda_, kappa, n_core) * alpha(lambda_, a)
    rho2 = NI_manual(lambda_, n_core) * alpha(lambda_, a)
    return NII(lambda_, kappa, n_core) * z1(j, rho1) * eta1(j, rho2) - NI_manual(lambda_, n_core) * eta1(j, rho1) * z1(j, rho2)

def KNR(j, lambda_, a, kappa, n_core):
    rho1 = NR(lambda_, kappa, n_core) * alpha(lambda_, a)
    rho2 = NI_manual(lambda_, n_core) * alpha(lambda_, a)
    return NI_manual(lambda_, n_core) * z1(j, rho1) * eta1(j, rho2) - NII(lambda_, kappa, n_core) * eta1(j, rho1) * z1(j, rho2)

def FNL(j, lambda_, a, kappa, n_core):
    rho1 = NL(lambda_, kappa, n_core) * alpha(lambda_, a)
    rho2 = NI_manual(lambda_, n_core) * alpha(lambda_, a)
    return NII(lambda_, kappa, n_core) * z2(j, rho1) * eta1(j, rho2) - NI_manual(lambda_, n_core) * eta2(j, rho1) * z1(j, rho2)

def GNL(j, lambda_, a, kappa, n_core):
    rho1 = NL(lambda_, kappa, n_core) * alpha(lambda_, a)
    rho2 = NI_manual(lambda_, n_core) * alpha(lambda_, a)
    return NI_manual(lambda_, n_core) * z2(j, rho1) * eta1(j, rho2) - NII(lambda_, kappa, n_core) * eta2(j, rho1) * z1(j, rho2)

def HNL(j, lambda_, a, kappa, n_core):
    rho1 = NL(lambda_, kappa, n_core) * alpha(lambda_, a)
    rho2 = NI_manual(lambda_, n_core) * alpha(lambda_, a)
    return NII(lambda_, kappa, n_core) * z1(j, rho1) * eta1(j, rho2) - NI_manual(lambda_, n_core) * eta1(j, rho1) * z1(j, rho2)

def KNL(j, lambda_, a, kappa, n_core):
    rho1 = NL(lambda_, kappa, n_core) * alpha(lambda_, a)
    rho2 = NI_manual(lambda_, n_core) * alpha(lambda_, a)
    return NI_manual(lambda_, n_core) * z1(j, rho1) * eta1(j, rho2) - NII(lambda_, kappa, n_core) * eta1(j, rho1) * z1(j, rho2)

#----------------------------------------------------------------------------------------------------------------------------------------------
# ------------------ Funciones derivadas DN1–DN4 ------------------

def DN(j, lambda_, a, kappa, n_core):
    return FNL(j, lambda_, a, kappa, n_core) * GNR(j, lambda_, a, kappa, n_core) + \
           FNR(j, lambda_, a, kappa, n_core) * GNL(j, lambda_, a, kappa, n_core)

def DN1(j, lambda_, a, kappa, n_core):
    return -1 / DN(j, lambda_, a, kappa, n_core) * (
        GNR(j, lambda_, a, kappa, n_core) * HNL(j, lambda_, a, kappa, n_core) +
        FNR(j, lambda_, a, kappa, n_core) * KNL(j, lambda_, a, kappa, n_core)
    )

def DN2(j, lambda_, a, kappa, n_core):
    return 1 / DN(j, lambda_, a, kappa, n_core) * (
        FNR(j, lambda_, a, kappa, n_core) * KNR(j, lambda_, a, kappa, n_core) -
        GNR(j, lambda_, a, kappa, n_core) * HNR(j, lambda_, a, kappa, n_core)
    )

def DN3(j, lambda_, a, kappa, n_core):
    return 1 / DN(j, lambda_, a, kappa, n_core) * (
        GNL(j, lambda_, a, kappa, n_core) * HNL(j, lambda_, a, kappa, n_core) -
        FNL(j, lambda_, a, kappa, n_core) * KNL(j, lambda_, a, kappa, n_core)
    )

def DN4(j, lambda_, a, kappa, n_core):
    return -1 / DN(j, lambda_, a, kappa, n_core) * (
        GNL(j, lambda_, a, kappa, n_core) * HNR(j, lambda_, a, kappa, n_core) +
        FNL(j, lambda_, a, kappa, n_core) * KNR(j, lambda_, a, kappa, n_core)
    )

# ------------------ Funciones XNL/XNR/UNL/UNR ± ------------------

def XNRplus(j, lambda_, a, b, kappa, n_core):
    return z1(j, NR(lambda_, kappa, n_core) * v(lambda_, b)) + \
           DN4(j, lambda_, a, kappa, n_core) * z2(j, NR(lambda_, kappa, n_core) * v(lambda_, b)) + \
           DN2(j, lambda_, a, kappa, n_core) * z2(j, NL(lambda_, kappa, n_core) * v(lambda_, b))

def XNRminus(j, lambda_, a, b, kappa, n_core):
    return z1(j, NR(lambda_, kappa, n_core) * v(lambda_, b)) + \
           DN4(j, lambda_, a, kappa, n_core) * z2(j, NR(lambda_, kappa, n_core) * v(lambda_, b)) - \
           DN2(j, lambda_, a, kappa, n_core) * z2(j, NL(lambda_, kappa, n_core) * v(lambda_, b))

def XNLplus(j, lambda_, a, b, kappa, n_core):
    return z1(j, NL(lambda_, kappa, n_core) * v(lambda_, b)) + \
           DN1(j, lambda_, a, kappa, n_core) * z2(j, NL(lambda_, kappa, n_core) * v(lambda_, b)) + \
           DN3(j, lambda_, a, kappa, n_core) * z2(j, NR(lambda_, kappa, n_core) * v(lambda_, b))

def XNLminus(j, lambda_, a, b, kappa, n_core):
    return z1(j, NL(lambda_, kappa, n_core) * v(lambda_, b)) + \
           DN1(j, lambda_, a, kappa, n_core) * z2(j, NL(lambda_, kappa, n_core) * v(lambda_, b)) - \
           DN3(j, lambda_, a, kappa, n_core) * z2(j, NR(lambda_, kappa, n_core) * v(lambda_, b))

def UNRplus(j, lambda_, a, b, kappa, n_core):
    return eta1(j, NR(lambda_, kappa, n_core) * v(lambda_, b)) + \
           DN4(j, lambda_, a, kappa, n_core) * eta2(j, NR(lambda_, kappa, n_core) * v(lambda_, b)) + \
           DN2(j, lambda_, a, kappa, n_core) * eta2(j, NL(lambda_, kappa, n_core) * v(lambda_, b))

def UNRminus(j, lambda_, a, b, kappa, n_core):
    return eta1(j, NR(lambda_, kappa, n_core) * v(lambda_, b)) + \
           DN4(j, lambda_, a, kappa, n_core) * eta2(j, NR(lambda_, kappa, n_core) * v(lambda_, b)) - \
           DN2(j, lambda_, a, kappa, n_core) * eta2(j, NL(lambda_, kappa, n_core) * v(lambda_, b))

def UNLplus(j, lambda_, a, b, kappa, n_core):
    return eta1(j, NL(lambda_, kappa, n_core) * v(lambda_, b)) + \
           DN1(j, lambda_, a, kappa, n_core) * eta2(j, NL(lambda_, kappa, n_core) * v(lambda_, b)) + \
           DN3(j, lambda_, a, kappa, n_core) * eta2(j, NR(lambda_, kappa, n_core) * v(lambda_, b))

def UNLminus(j, lambda_, a, b, kappa, n_core):
    return eta1(j, NL(lambda_, kappa, n_core) * v(lambda_, b)) + \
           DN1(j, lambda_, a, kappa, n_core) * eta2(j, NL(lambda_, kappa, n_core) * v(lambda_, b)) - \
           DN3(j, lambda_, a, kappa, n_core) * eta2(j, NR(lambda_, kappa, n_core) * v(lambda_, b))

# ------------------ Funciones AN, BN, VN, WN ------------------

def ANR(j, lambda_, a, b, kappa, n_core):
    return XNRminus(j, lambda_, a, b, kappa, n_core) * eta1(j, v(lambda_, b)) - \
           NII(lambda_, kappa, n_core) * UNRminus(j, lambda_, a, b, kappa, n_core) * z1(j, v(lambda_, b))

def ANL(j, lambda_, a, b, kappa, n_core):
    return XNLplus(j, lambda_, a, b, kappa, n_core) * eta1(j, v(lambda_, b)) - \
           NII(lambda_, kappa, n_core) * UNLplus(j, lambda_, a, b, kappa, n_core) * z1(j, v(lambda_, b))

def WNL(j, lambda_, a, b, kappa, n_core):
    return XNLminus(j, lambda_, a, b, kappa, n_core) * NII(lambda_, kappa, n_core) * eta3(j, v(lambda_, b)) - \
           UNLminus(j, lambda_, a, b, kappa, n_core) * z3(j, v(lambda_, b))

def WNR(j, lambda_, a, b, kappa, n_core):
    return XNRplus(j, lambda_, a, b, kappa, n_core) * NII(lambda_, kappa, n_core) * eta3(j, v(lambda_, b)) - \
           UNRplus(j, lambda_, a, b, kappa, n_core) * z3(j, v(lambda_, b))

def VNL(j, lambda_, a, b, kappa, n_core):
    return XNLplus(j, lambda_, a, b, kappa, n_core) * eta3(j, v(lambda_, b)) - \
           NII(lambda_, kappa, n_core) * UNLplus(j, lambda_, a, b, kappa, n_core) * z3(j, v(lambda_, b))

def VNR(j, lambda_, a, b, kappa, n_core):
    return XNRminus(j, lambda_, a, b, kappa, n_core) * eta3(j, v(lambda_, b)) - \
           NII(lambda_, kappa, n_core) * UNRminus(j, lambda_, a, b, kappa, n_core) * z3(j, v(lambda_, b))

def BNL(j, lambda_, a, b, kappa, n_core):
    return XNLminus(j, lambda_, a, b, kappa, n_core) * NII(lambda_, kappa, n_core) * eta1(j, v(lambda_, b)) - \
           UNLminus(j, lambda_, a, b, kappa, n_core) * z1(j, v(lambda_, b))

def BNR(j, lambda_, a, b, kappa, n_core):
    return XNRplus(j, lambda_, a, b, kappa, n_core) * NII(lambda_, kappa, n_core) * eta1(j, v(lambda_, b)) - \
           UNRplus(j, lambda_, a, b, kappa, n_core) * z1(j, v(lambda_, b))

def Delta(j, lambda_, a, b, kappa, n_core):
    return WNL(j, lambda_, a, b, kappa, n_core) * VNR(j, lambda_, a, b, kappa, n_core) + \
           WNR(j, lambda_, a, b, kappa, n_core) * VNL(j, lambda_, a, b, kappa, n_core)
#----------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------- Coeficientes de Mie -----------------------------

def an(j, lambda_, a, b, kappa, n_core):
    return 1 / Delta(j, lambda_, a, b, kappa, n_core) * (
        BNL(j, lambda_, a, b, kappa, n_core) * VNR(j, lambda_, a, b, kappa, n_core) +
        BNR(j, lambda_, a, b, kappa, n_core) * VNL(j, lambda_, a, b, kappa, n_core)
    )

def bn(j, lambda_, a, b, kappa, n_core):
    return 1 / Delta(j, lambda_, a, b, kappa, n_core) * (
        ANR(j, lambda_, a, b, kappa, n_core) * WNL(j, lambda_, a, b, kappa, n_core) +
        ANL(j, lambda_, a, b, kappa, n_core) * WNR(j, lambda_, a, b, kappa, n_core)
    )

def cn(j, lambda_, a, b, kappa, n_core):
    return 1j / Delta(j, lambda_, a, b, kappa, n_core) * (
        ANL(j, lambda_, a, b, kappa, n_core) * VNR(j, lambda_, a, b, kappa, n_core) -
        ANR(j, lambda_, a, b, kappa, n_core) * VNL(j, lambda_, a, b, kappa, n_core)
    )

def dn(j, lambda_, a, b, kappa, n_core):
    return 1j / Delta(j, lambda_, a, b, kappa, n_core) * (
        BNR(j, lambda_, a, b, kappa, n_core) * WNL(j, lambda_, a, b, kappa, n_core) -
        BNL(j, lambda_, a, b, kappa, n_core) * WNR(j, lambda_, a, b, kappa, n_core)
    )

# ----------------------------- Parámetros derivados -----------------------------

def m_2(kappa, n_core):
    return 0.5 * (n_sL(kappa, n_core) + n_sR(kappa, n_core))

def Largest(lambda_, r, kappa):
    val = v(lambda_, r)
    return max(val, abs(val), abs(val * m_2(kappa, n_core)))

def LastTerm(r, kappa):
    lambda0 = 1064 * nm  # Asumido valor fijo de lambda₀ como en nI()
    L = Largest(lambda0, r, kappa)
    return int(np.ceil(abs(L + 4.05 * L**(1 / 3) + 2)))

# ----------------------------- Transformación final -----------------------------

def An(j, lambda_, a, r, kappa, n_core):
    return an(j, lambda_, a, r, kappa, n_core) + sigma * 1j * dn(j, lambda_, a, r, kappa, n_core)

def Bn(j, lambda_, a, r, kappa, n_core):
    return bn(j, lambda_, a, r, kappa, n_core) - sigma * 1j * cn(j, lambda_, a, r, kappa, n_core)
#---------------------------------------------------------------------------------------------------------------------------------------------
# Último término a usar en las sumas (puede ser redefinido dinámicamente con LastTerm)
lastterm = 4

# ----------------------------- Eficiencias -----------------------------

def Qsca(lambda_, a, x, kappa, n_core):
    r = x / k_0(lambda_)  # radio exterior
    result = 0
    for j in range(1, lastterm + 1):
        A = An(j, lambda_, a, r, kappa, n_core)
        B = Bn(j, lambda_, a, r, kappa, n_core)
        result += (2 * j + 1) * (abs(A)**2 + abs(B)**2)
    return 2 / x**2 * result


def Qext(lambda_, a, x, kappa, n_core):
    r = x / k_0(lambda_)
    result = 0
    for j in range(1, lastterm + 1):
        A = An(j, lambda_, a, r, kappa, n_core)
        B = Bn(j, lambda_, a, r, kappa, n_core)
        result += (2 * j + 1) * (An(j, lambda_, a, r, kappa, n_core) + Bn(j, lambda_, a, r, kappa, n_core))
    return 2 / x**2 * np.real(result)

def Qabs(lambda_, a, r, kappa, n_core):
    return Qext(lambda_, a, r, kappa, n_core) - Qsca(lambda_, a, r, kappa, n_core)

# ----------------------------- Asimetría (g-factor) -----------------------------

def QacCos(lambda_, a, r, kappa, n_core):
    result = 0
    for j in range(1, lastterm + 1):
        # Primer término cruzado con j+1
        if j + 1 <= lastterm:
            cross = (
                An(j, lambda_, a, r, kappa, n_core) * np.conj(An(j + 1, lambda_, a, r, kappa, n_core)) +
                Bn(j, lambda_, a, r, kappa, n_core) * np.conj(Bn(j + 1, lambda_, a, r, kappa, n_core))
            )
            result += (j * (j + 2)) / (j + 1) * np.real(cross)
        # Segundo término cruzado dentro del mismo j
        cross2 = An(j, lambda_, a, r, kappa, n_core) * np.conj(Bn(j, lambda_, a, r, kappa, n_core))
        result += (2 * j + 1) / (j * (j + 1)) * np.real(cross2)
    return 4 / v(lambda_, r)**2 * result

def gfactor(lambda_, a, r, kappa, n_core):
    return QacCos(lambda_, a, r, kappa, n_core) / Qsca(lambda_, a, r, kappa, n_core)

# ----------------------------- Fuerza óptica (extensión) -----------------------------

def Force(lambda_, a, r, kappa, n_core):
    return Qext(lambda_, a, r, kappa, n_core) - QacCos(lambda_, a, r, kappa, n_core)
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#Gráficas
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# Parámetros fijos
lambda_vals_nm = np.linspace(300, 1000, 150)  # en nm
a = 150e-9
r_vals_m = np.linspace(50e-9, 150e-9, 150)    # en metros

# Rango de valores para kappa


# Crear malla 2D
Lambda, R = np.meshgrid(lambda_vals_nm, r_vals_m)
kappa = 0

Z_sca = np.zeros_like(Lambda)
Z_ext = np.zeros_like(Lambda)


for i in range(Lambda.shape[0]):
    for j in range(Lambda.shape[1]):
        lambda_val_m = Lambda[i, j] * 1e-9  # convierte nm a m
        r_val = R[i, j]
        n_core_val = n_complex(Lambda[i, j])  # índice complejo
        x_val = r_val * k_0(lambda_val_m)
        Z_sca[i, j] = Qsca(lambda_val_m, a, x_val, kappa, n_core_val)
        Z_ext[i, j] = Qext(lambda_val_m, a, x_val, kappa, n_core_val)
R_nm = R * 1e9  # pasa R a nanómetros
# Graficar ambos en subplots 3D
# Calcular escala común de colores
z_min = min(Z_sca.min(), Z_ext.min())
z_max = max(Z_sca.max(), Z_ext.max())

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Qsca
im1 = axs[0].imshow(Z_sca, origin='lower', aspect='auto',
                    extent=[lambda_vals_nm.min(), lambda_vals_nm.max(), R_nm.min(), R_nm.max()],
                    cmap='viridis', vmin=z_min, vmax=z_max)
axs[0].set_xlabel('Longitud de onda (nm)')
axs[0].set_ylabel('Radio de la esfera (nm)')
axs[0].set_title('Qsca vs r y $\lambda$ con $\sigma=1$')

# Qext
im2 = axs[1].imshow(Z_ext, origin='lower', aspect='auto',
                    extent=[lambda_vals_nm.min(), lambda_vals_nm.max(), R_nm.min(), R_nm.max()],
                    cmap='viridis', vmin=z_min, vmax=z_max)
axs[1].set_xlabel('Longitud de onda (nm)')
axs[1].set_ylabel('Radio de la esfera (nm)')
axs[1].set_title('Qext vs r y $\lambda$ con $\sigma=1$')

# Barra de color vertical fina al lado derecho
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
fig.colorbar(im2, cax=cbar_ax, orientation='vertical', label='Eficiencia (Q)')

plt.tight_layout(rect=[0, 0, 0.9, 1])  # deja espacio a la derecha para la barra
plt.show()


