#Este código representa las eficiencias de scattering y extinción frente al parámetro de
#quiralidad kappa con un core-shell quiral y sin absorción (eh es real).
#hay que cambiar la helicidad manualmente buscando "sigma" en el buscador. Recordar cambiarlo también en los títulos de las gráficas.

import numpy as np
from scipy.special import spherical_jn as besselj, spherical_yn as bessely
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

# Constantes electromagnéticas
epsilon_0 = 8.85e-12
mu_0 = 4 * np.pi * 1e-7  # H/m
eta = 0.48

# Cálculo de sigma_b
sigma_b = (6 * np.pi * epsilon_0 * c**3 * eta) / (
    tau_21 * (2 * np.pi * c / lambda_a)**2 * np.sqrt(eh)
)

# Función de ganancia Gainm
def Gainm(lambda_, Gamma_pump, mc):
    omega = 2 * np.pi * c / lambda_
    omega_a = 2 * np.pi * c / lambda_a
    denom = omega**2 + 1j * Delta_Omega_a * omega - omega_a**2
    num = ((tau_21 - tau_10) * Gamma_pump) / (
        1 + (tau_32 + tau_21 + tau_10) * Gamma_pump
    ) * mc
    eps_eff = epsilon_0 * eh + (sigma_b / denom) * num
    return np.sqrt(eps_eff / epsilon_0)

#----------------------------------------------------------------------------------------------------------------------------------------------------
# Parámetros ópticos y constantes adicionales
sigma = 1  # Polarización circular

# Permisividades y permeabilidades
epsilon_1 = 1.332**2  # Permisividad del medio
mu_1 = 1  # Permeabilidad del medio
epsilon_sh = 2.89  # Permisividad de la cubierta (shell)
mu_sh = 1  # Permeabilidad de la cubierta (shell)
mu_c = 1  # Permeabilidad del núcleo
n_1 = np.sqrt(epsilon_1)  # Índice de refracción del medio

# Índices de refracción relativos del shell
def n_sL(kappa):
    return np.sqrt(epsilon_sh * mu_sh) + kappa

def n_sR(kappa):
    return np.sqrt(epsilon_sh * mu_sh) - kappa

# Índice de refracción del núcleo (material con ganancia)
def nI(mc):
    return Gainm(1064 * nm, mc, 5e24)

# Función m₂ (media entre n_sL y n_sR)
def m_2(kappa):
    return 0.5 * (n_sL(kappa) + n_sR(kappa))

# Vector de onda y parámetros relacionados
def K_1(lambda_):
    return n_1 * (2 * np.pi / lambda_)

def k_0(lambda_):
    return n_1 * (2 * np.pi / lambda_)

def k_LL(lambda_, kappa):
    return n_sL(kappa) * (2 * np.pi / lambda_)

def k_R(lambda_, kappa):
    return n_sR(kappa) * (2 * np.pi / lambda_)

def k_II(lambda_, kappa):
    return 0.5 * (k_LL(lambda_, kappa) + k_R(lambda_, kappa))

def k_I(lambda_, mc):
    return nI(mc) * (2 * np.pi / lambda_)

# Parámetro de tamaño en agua
def q_1(lambda_, r):
    return K_1(lambda_) * r
#----------------------------------------------------------------------------------------------------------------------------------------------------
def x_0(lambda_, a):
    return n_1 * k_0(lambda_) * a

def v(lambda_, b):
    return k_0(lambda_) * b

def alpha(lambda_, a):
    return k_0(lambda_) * a

def NL(lambda_, kappa):
    return k_LL(lambda_, kappa) / k_0(lambda_)

def NR(lambda_, kappa):
    return k_R(lambda_, kappa) / k_0(lambda_)

def NII(lambda_, kappa):
    return k_II(lambda_, kappa) / k_0(lambda_)

def NI(lambda_, mc):
    return k_I(lambda_, mc) / k_0(lambda_)

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

def FNR(j, lambda_, a, kappa, mc):
    rho1 = NR(lambda_, kappa) * alpha(lambda_, a)
    rho2 = NI(lambda_, mc) * alpha(lambda_, a)
    return NII(lambda_, kappa) * z2(j, rho1) * eta1(j, rho2) - NI(lambda_, mc) * eta2(j, rho1) * z1(j, rho2)

def GNR(j, lambda_, a, kappa, mc):
    rho1 = NR(lambda_, kappa) * alpha(lambda_, a)
    rho2 = NI(lambda_, mc) * alpha(lambda_, a)
    return NI(lambda_, mc) * z2(j, rho1) * eta1(j, rho2) - NII(lambda_, kappa) * eta2(j, rho1) * z1(j, rho2)

def HNR(j, lambda_, a, kappa, mc):
    rho1 = NR(lambda_, kappa) * alpha(lambda_, a)
    rho2 = NI(lambda_, mc) * alpha(lambda_, a)
    return NII(lambda_, kappa) * z1(j, rho1) * eta1(j, rho2) - NI(lambda_, mc) * eta1(j, rho1) * z1(j, rho2)

def KNR(j, lambda_, a, kappa, mc):
    rho1 = NR(lambda_, kappa) * alpha(lambda_, a)
    rho2 = NI(lambda_, mc) * alpha(lambda_, a)
    return NI(lambda_, mc) * z1(j, rho1) * eta1(j, rho2) - NII(lambda_, kappa) * eta1(j, rho1) * z1(j, rho2)

def FNL(j, lambda_, a, kappa, mc):
    rho1 = NL(lambda_, kappa) * alpha(lambda_, a)
    rho2 = NI(lambda_, mc) * alpha(lambda_, a)
    return NII(lambda_, kappa) * z2(j, rho1) * eta1(j, rho2) - NI(lambda_, mc) * eta2(j, rho1) * z1(j, rho2)

def GNL(j, lambda_, a, kappa, mc):
    rho1 = NL(lambda_, kappa) * alpha(lambda_, a)
    rho2 = NI(lambda_, mc) * alpha(lambda_, a)
    return NI(lambda_, mc) * z2(j, rho1) * eta1(j, rho2) - NII(lambda_, kappa) * eta2(j, rho1) * z1(j, rho2)

def HNL(j, lambda_, a, kappa, mc):
    rho1 = NL(lambda_, kappa) * alpha(lambda_, a)
    rho2 = NI(lambda_, mc) * alpha(lambda_, a)
    return NII(lambda_, kappa) * z1(j, rho1) * eta1(j, rho2) - NI(lambda_, mc) * eta1(j, rho1) * z1(j, rho2)

def KNL(j, lambda_, a, kappa, mc):
    rho1 = NL(lambda_, kappa) * alpha(lambda_, a)
    rho2 = NI(lambda_, mc) * alpha(lambda_, a)
    return NI(lambda_, mc) * z1(j, rho1) * eta1(j, rho2) - NII(lambda_, kappa) * eta1(j, rho1) * z1(j, rho2)

#----------------------------------------------------------------------------------------------------------------------------------------------
# ------------------ Funciones derivadas DN1–DN4 ------------------

def DN(j, lambda_, a, kappa, mc):
    return FNL(j, lambda_, a, kappa, mc) * GNR(j, lambda_, a, kappa, mc) + \
           FNR(j, lambda_, a, kappa, mc) * GNL(j, lambda_, a, kappa, mc)

def DN1(j, lambda_, a, kappa, mc):
    return -1 / DN(j, lambda_, a, kappa, mc) * (
        GNR(j, lambda_, a, kappa, mc) * HNL(j, lambda_, a, kappa, mc) +
        FNR(j, lambda_, a, kappa, mc) * KNL(j, lambda_, a, kappa, mc)
    )

def DN2(j, lambda_, a, kappa, mc):
    return 1 / DN(j, lambda_, a, kappa, mc) * (
        FNR(j, lambda_, a, kappa, mc) * KNR(j, lambda_, a, kappa, mc) -
        GNR(j, lambda_, a, kappa, mc) * HNR(j, lambda_, a, kappa, mc)
    )

def DN3(j, lambda_, a, kappa, mc):
    return 1 / DN(j, lambda_, a, kappa, mc) * (
        GNL(j, lambda_, a, kappa, mc) * HNL(j, lambda_, a, kappa, mc) -
        FNL(j, lambda_, a, kappa, mc) * KNL(j, lambda_, a, kappa, mc)
    )

def DN4(j, lambda_, a, kappa, mc):
    return -1 / DN(j, lambda_, a, kappa, mc) * (
        GNL(j, lambda_, a, kappa, mc) * HNR(j, lambda_, a, kappa, mc) +
        FNL(j, lambda_, a, kappa, mc) * KNR(j, lambda_, a, kappa, mc)
    )

# ------------------ Funciones XNL/XNR/UNL/UNR ± ------------------

def XNRplus(j, lambda_, a, b, kappa, mc):
    return z1(j, NR(lambda_, kappa) * v(lambda_, b)) + \
           DN4(j, lambda_, a, kappa, mc) * z2(j, NR(lambda_, kappa) * v(lambda_, b)) + \
           DN2(j, lambda_, a, kappa, mc) * z2(j, NL(lambda_, kappa) * v(lambda_, b))

def XNRminus(j, lambda_, a, b, kappa, mc):
    return z1(j, NR(lambda_, kappa) * v(lambda_, b)) + \
           DN4(j, lambda_, a, kappa, mc) * z2(j, NR(lambda_, kappa) * v(lambda_, b)) - \
           DN2(j, lambda_, a, kappa, mc) * z2(j, NL(lambda_, kappa) * v(lambda_, b))

def XNLplus(j, lambda_, a, b, kappa, mc):
    return z1(j, NL(lambda_, kappa) * v(lambda_, b)) + \
           DN1(j, lambda_, a, kappa, mc) * z2(j, NL(lambda_, kappa) * v(lambda_, b)) + \
           DN3(j, lambda_, a, kappa, mc) * z2(j, NR(lambda_, kappa) * v(lambda_, b))

def XNLminus(j, lambda_, a, b, kappa, mc):
    return z1(j, NL(lambda_, kappa) * v(lambda_, b)) + \
           DN1(j, lambda_, a, kappa, mc) * z2(j, NL(lambda_, kappa) * v(lambda_, b)) - \
           DN3(j, lambda_, a, kappa, mc) * z2(j, NR(lambda_, kappa) * v(lambda_, b))

def UNRplus(j, lambda_, a, b, kappa, mc):
    return eta1(j, NR(lambda_, kappa) * v(lambda_, b)) + \
           DN4(j, lambda_, a, kappa, mc) * eta2(j, NR(lambda_, kappa) * v(lambda_, b)) + \
           DN2(j, lambda_, a, kappa, mc) * eta2(j, NL(lambda_, kappa) * v(lambda_, b))

def UNRminus(j, lambda_, a, b, kappa, mc):
    return eta1(j, NR(lambda_, kappa) * v(lambda_, b)) + \
           DN4(j, lambda_, a, kappa, mc) * eta2(j, NR(lambda_, kappa) * v(lambda_, b)) - \
           DN2(j, lambda_, a, kappa, mc) * eta2(j, NL(lambda_, kappa) * v(lambda_, b))

def UNLplus(j, lambda_, a, b, kappa, mc):
    return eta1(j, NL(lambda_, kappa) * v(lambda_, b)) + \
           DN1(j, lambda_, a, kappa, mc) * eta2(j, NL(lambda_, kappa) * v(lambda_, b)) + \
           DN3(j, lambda_, a, kappa, mc) * eta2(j, NR(lambda_, kappa) * v(lambda_, b))

def UNLminus(j, lambda_, a, b, kappa, mc):
    return eta1(j, NL(lambda_, kappa) * v(lambda_, b)) + \
           DN1(j, lambda_, a, kappa, mc) * eta2(j, NL(lambda_, kappa) * v(lambda_, b)) - \
           DN3(j, lambda_, a, kappa, mc) * eta2(j, NR(lambda_, kappa) * v(lambda_, b))

# ------------------ Funciones AN, BN, VN, WN ------------------

def ANR(j, lambda_, a, b, kappa, mc):
    return XNRminus(j, lambda_, a, b, kappa, mc) * eta1(j, v(lambda_, b)) - \
           NII(lambda_, kappa) * UNRminus(j, lambda_, a, b, kappa, mc) * z1(j, v(lambda_, b))

def ANL(j, lambda_, a, b, kappa, mc):
    return XNLplus(j, lambda_, a, b, kappa, mc) * eta1(j, v(lambda_, b)) - \
           NII(lambda_, kappa) * UNLplus(j, lambda_, a, b, kappa, mc) * z1(j, v(lambda_, b))

def WNL(j, lambda_, a, b, kappa, mc):
    return XNLminus(j, lambda_, a, b, kappa, mc) * NII(lambda_, kappa) * eta3(j, v(lambda_, b)) - \
           UNLminus(j, lambda_, a, b, kappa, mc) * z3(j, v(lambda_, b))

def WNR(j, lambda_, a, b, kappa, mc):
    return XNRplus(j, lambda_, a, b, kappa, mc) * NII(lambda_, kappa) * eta3(j, v(lambda_, b)) - \
           UNRplus(j, lambda_, a, b, kappa, mc) * z3(j, v(lambda_, b))

def VNL(j, lambda_, a, b, kappa, mc):
    return XNLplus(j, lambda_, a, b, kappa, mc) * eta3(j, v(lambda_, b)) - \
           NII(lambda_, kappa) * UNLplus(j, lambda_, a, b, kappa, mc) * z3(j, v(lambda_, b))

def VNR(j, lambda_, a, b, kappa, mc):
    return XNRminus(j, lambda_, a, b, kappa, mc) * eta3(j, v(lambda_, b)) - \
           NII(lambda_, kappa) * UNRminus(j, lambda_, a, b, kappa, mc) * z3(j, v(lambda_, b))

def BNL(j, lambda_, a, b, kappa, mc):
    return XNLminus(j, lambda_, a, b, kappa, mc) * NII(lambda_, kappa) * eta1(j, v(lambda_, b)) - \
           UNLminus(j, lambda_, a, b, kappa, mc) * z1(j, v(lambda_, b))

def BNR(j, lambda_, a, b, kappa, mc):
    return XNRplus(j, lambda_, a, b, kappa, mc) * NII(lambda_, kappa) * eta1(j, v(lambda_, b)) - \
           UNRplus(j, lambda_, a, b, kappa, mc) * z1(j, v(lambda_, b))

def Delta(j, lambda_, a, b, kappa, mc):
    return WNL(j, lambda_, a, b, kappa, mc) * VNR(j, lambda_, a, b, kappa, mc) + \
           WNR(j, lambda_, a, b, kappa, mc) * VNL(j, lambda_, a, b, kappa, mc)
#----------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------- Coeficientes de Mie -----------------------------

def an(j, lambda_, a, b, kappa, mc):
    return 1 / Delta(j, lambda_, a, b, kappa, mc) * (
        BNL(j, lambda_, a, b, kappa, mc) * VNR(j, lambda_, a, b, kappa, mc) +
        BNR(j, lambda_, a, b, kappa, mc) * VNL(j, lambda_, a, b, kappa, mc)
    )

def bn(j, lambda_, a, b, kappa, mc):
    return 1 / Delta(j, lambda_, a, b, kappa, mc) * (
        ANR(j, lambda_, a, b, kappa, mc) * WNL(j, lambda_, a, b, kappa, mc) +
        ANL(j, lambda_, a, b, kappa, mc) * WNR(j, lambda_, a, b, kappa, mc)
    )

def cn(j, lambda_, a, b, kappa, mc):
    return 1j / Delta(j, lambda_, a, b, kappa, mc) * (
        ANL(j, lambda_, a, b, kappa, mc) * VNR(j, lambda_, a, b, kappa, mc) -
        ANR(j, lambda_, a, b, kappa, mc) * VNL(j, lambda_, a, b, kappa, mc)
    )

def dn(j, lambda_, a, b, kappa, mc):
    return 1j / Delta(j, lambda_, a, b, kappa, mc) * (
        BNR(j, lambda_, a, b, kappa, mc) * WNL(j, lambda_, a, b, kappa, mc) -
        BNL(j, lambda_, a, b, kappa, mc) * WNR(j, lambda_, a, b, kappa, mc)
    )

# ----------------------------- Parámetros derivados -----------------------------

def m_2(kappa):
    return 0.5 * (n_sL(kappa) + n_sR(kappa))

def Largest(lambda_, r, kappa):
    val = v(lambda_, r)
    return max(val, abs(val), abs(val * m_2(kappa)))

def LastTerm(r, kappa):
    lambda0 = 1064 * nm  # Asumido valor fijo de lambda₀ como en nI()
    L = Largest(lambda0, r, kappa)
    return int(np.ceil(abs(L + 4.05 * L**(1 / 3) + 2)))

# ----------------------------- Transformación final -----------------------------

def An(j, lambda_, a, r, kappa, mc):
    return an(j, lambda_, a, r, kappa, mc) + sigma * 1j * dn(j, lambda_, a, r, kappa, mc)

def Bn(j, lambda_, a, r, kappa, mc):
    return bn(j, lambda_, a, r, kappa, mc) - sigma * 1j * cn(j, lambda_, a, r, kappa, mc)
#---------------------------------------------------------------------------------------------------------------------------------------------
# Último término a usar en las sumas (puede ser redefinido dinámicamente con LastTerm)
lastterm = 4

# ----------------------------- Eficiencias -----------------------------

def Qsca(lambda_, a, r, kappa, mc):
    result = 0
    for j in range(1, lastterm + 1):
        term = (2 * j + 1) * (abs(An(j, lambda_, a, r, kappa, mc))**2 + abs(Bn(j, lambda_, a, r, kappa, mc))**2)
        result += term
    return 2 / v(lambda_, r)**2 * result


def Qext(lambda_, a, r, kappa, mc):
    result = 0
    for j in range(1, lastterm + 1):
        term = (2 * j + 1) * (An(j, lambda_, a, r, kappa, mc) + Bn(j, lambda_, a, r, kappa, mc))
        result += term
    return 2 / v(lambda_, r)**2 * np.real(result)

def Qabs(lambda_, a, r, kappa, mc):
    return Qext(lambda_, a, r, kappa, mc) - Qsca(lambda_, a, r, kappa, mc)

# ----------------------------- Asimetría (g-factor) -----------------------------

def QacCos(lambda_, a, r, kappa, mc):
    result = 0
    for j in range(1, lastterm + 1):
        # Primer término cruzado con j+1
        if j + 1 <= lastterm:
            cross = (
                An(j, lambda_, a, r, kappa, mc) * np.conj(An(j + 1, lambda_, a, r, kappa, mc)) +
                Bn(j, lambda_, a, r, kappa, mc) * np.conj(Bn(j + 1, lambda_, a, r, kappa, mc))
            )
            result += (j * (j + 2)) / (j + 1) * np.real(cross)
        # Segundo término cruzado dentro del mismo j
        cross2 = An(j, lambda_, a, r, kappa, mc) * np.conj(Bn(j, lambda_, a, r, kappa, mc))
        result += (2 * j + 1) / (j * (j + 1)) * np.real(cross2)
    return 4 / v(lambda_, r)**2 * result

def gfactor(lambda_, a, r, kappa, mc):
    return QacCos(lambda_, a, r, kappa, mc) / Qsca(lambda_, a, r, kappa, mc)

# ----------------------------- Fuerza óptica (extensión) -----------------------------

def Force(lambda_, a, r, kappa, mc):
    return Qext(lambda_, a, r, kappa, mc) - QacCos(lambda_, a, r, kappa, mc)
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#Gráficas
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# Parámetros fijos
lambda_val = 1064e-9  # en nm
a = 200e-9
r = 280e-9
mc = 0 * 1e10

# Rango de valores para kappa
kappa_values = np.linspace(-1, 1, 81)  # 81 puntos incluyendo -1 y 1


# Evaluar Qsca y Qext para cada valor de kappa
table = [(kappa, Qsca(lambda_val , a , r , kappa, mc)) for kappa in kappa_values]
table1 = [(kappa, Qext(lambda_val , a , r , kappa, mc)) for kappa in kappa_values]
# Suponemos que 'table' ya contiene los datos [(kappa, Qsca), ...]

# Separar los valores de kappa, Qsca y Qext
kappa_vals, qsca_vals = zip(*table)
kappa_vals1, qext_vals = zip(*table1)

# Graficar
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("$Q_{s}$ y $Q_{ext}$ vs $\\kappa$ ($\\sigma=1$, $\\lambda=1064$ nm y núcleo sin absorción)", fontsize=14)
# Gráfico 1: Qs vs kappa
axs[0].plot(kappa_vals, qsca_vals, marker='o', color='mediumseagreen', linestyle='-', linewidth=1)
axs[0].set_xlabel("$\\kappa$")
axs[0].set_ylabel("$Q_{s}$")
axs[0].set_ylim([-0.20, 3.5])
axs[0].grid(True)

# Gráfico 2: Qext vs kappa
axs[1].plot(kappa_vals, qext_vals, marker='s', color='c', linestyle='--', linewidth=1)
axs[1].set_xlabel("$\\kappa$")
axs[1].set_ylabel("$Q_{ext}$")
axs[1].set_ylim([-0.20, 3.5])
axs[1].grid(True)

plt.tight_layout()
plt.show()
