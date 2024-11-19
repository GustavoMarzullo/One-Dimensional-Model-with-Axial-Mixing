import math
import numpy as np
from Reator.conversao import rN2

def k(T:float):
    """
    Função para calcular as constantes cinéticas.
    Fonte: Bischoff (1979)

    Parameters
    ----------
    T
        Temperatura [K].
    Returns
    -------
    k1, k2
    """
    R = 8.314
    k1 = 1.79e4*np.exp(-20800/(R*T))
    k2 = 2.57e16*np.exp(-47400/(R*T))

    return k1, k2

def raρb(CT:np.array, T:float , P:float):
    """
    Função para calcular a taxa de reação.
    Fonte: Bischoff (1979)

    Parameters
    ----------
    CT
        Concentração molar  no reator (N2, H2, NH3) [mol/s].
    T
        Temperatura [K].
    P
        Pressão [atm].
    Returns
    -------
    Taxa de reação. [mol/s]
    """
    k1, k2 = k(T)
    f = 0.7 #fator de efetividade

    Y = CT/sum(CT) #fração molar
    p = Y*P #pressão parcial
    pN2, pH2, pNH3 = p

    result = (f*(k1*(pN2*pH2**1.5)/pNH3 - k2*pNH3/pH2**1.5))/3.6
    return result


#vazão de entrada
FT = 6000 #mol/s
F0N2=0.21825*FT
F0H2=0.65475*FT
F0NH3=0.05*FT
F0 =np.array([F0N2, F0H2, F0NH3])
Q = 2.5 #m³/s
C0 = F0/Q
C0N2, C0H2, C0NH3 = C0

Tin = 360+273.15 #ºC
Tr = 298.15 #ºC
Pin = 155 #atm

ε = 0.4
Ac = 7 #m²
U = 5 #W/(m².K)
d = 1 #m
ρb = 1816.5 #kg/m³

Dea = 1e-4 #m²/s
λea = 4e-3 
Cp = 1200 #J/(kg.ºC)
ΔHr = -111370 #J/mol_N2

ρG = 23 #kg/m³
v = Q/Ac #m³/s -> m/s
us = v*ε #velocidade superficial do gás

L0 = 0 #valor inicial de comprimento
LF = 5 #valor final de comprimento
N = 3 #número de espaços
h = (LF-L0)/N #step size

A = (ε*Dea)/(h**2) - us/(2*h)
B = (-2*ε*Dea)/(h**2)
C = (ε*Dea)/(h**2) + us/(2*h)
D = (3*ε*Dea)/(2*h) + us
E = (-2*ε*Dea)/(h)
F = (ε*Dea)/(2*h)
G = (λea)/(h**2) - (ρG*us*Cp)/(2*h)
H = (-2*λea)/(h**2) - 4*U/d
I = (λea)/(h**2) + (ρG*us*Cp)/(2*h)
J = (3*λea)/(2*h) + (ρG*us*Cp)
K = (-2*λea)/(h)
L =(λea)/(2*h)

#estimativas iniciais
CN2 = np.linspace(0.8*C0N2, 0.5*C0N2, N + 1)
T = np.linspace(Tin, Tr, N + 1)

for iteration in range(1000): 
    CN2_new = np.copy(CN2)
    T_new = np.copy(T)

    #calculando as concentrações
    CH2 = C0H2 - 3*(C0N2-CN2_new)
    CNH3 = C0NH3 + 2*(C0N2-CN2_new)
    CT = np.array([CN2_new, CH2, CNH3])

    #condições de contorno no início
    CN2_new[0] = (us*C0N2 - E*CN2[1] - F*CN2[2])/D
    T_new[0] = (ρG*us*Cp*Tin - K*T[1] - L*T[2])/J
    
    for i in range(1, N):
        _rN2 = rN2(CT[:,i], T[i], Pin)
        CN2_new[i] = (_rN2*ρb - A*CN2[i+1] - C*CN2[i-1])/B
        T_new[i] = (-(-ΔHr)*_rN2*ρb +4*U/d*Tr - G*T[i+1] - I*T[i-1])/H

    #condições de contorno no final
    CN2_new[N] = (4*CN2[N-1] - CN2[N-2])/3
    T_new[N] = (4*T[N-1] - T[N-2])/3

    # Checar por convergência
    if np.allclose(CN2, CN2_new, atol=1e-3):
        print("Sucesso na convergência!")
        break
    CN2 = CN2_new
    T = T_new

print(CN2.round(0))
print(T.round(0)-273)