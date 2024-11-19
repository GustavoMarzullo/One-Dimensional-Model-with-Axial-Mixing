import numpy as np
from scipy.optimize import fsolve, root
from Reator.conversao import rN2
import matplotlib.pyplot as plt

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
N = 10 #número de espaços
h = (LF-L0)/N #step size
L_eval = np.linspace(L0, LF, N+1)

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
CN2_est = np.linspace(C0N2, 0.75*C0N2, N + 1)
T_est = np.linspace(Tin, Tin, N + 1)
estimativa = np.concatenate([CN2_est, T_est])


def fobj(vars):
    #criando os vetores    
    CN2 = vars[:N+1]
    T = vars[N+1:]
    res = np.zeros(len(vars))

    #calculando as concentrações
    CH2 = C0H2 - 3*(C0N2-CN2)
    CNH3 = C0NH3 + 2*(C0N2-CN2)
    CT = np.array([CN2, CH2, CNH3])
    
    #condições de contorno em z=0
    res[0] = D*CN2[0] + E*CN2[1] + F*CN2[2] -us*C0N2 #vazão molar
    res[N+1] = J*T[0] + K*T[1] + L*T[2] - ρG*us*Cp*Tin #temperatura

    #pontos de dentro
    for i in range(1, N):
        _rN2 = -rN2(CT[:,i], T[i], Pin)
        res[i] =  A*CN2[i+1] + B*CN2[i] + C*CN2[i-1] - _rN2*ρb #vazão molar
        res[N+1+i] = G*T[i+1] + H*T[i] + I*T[i-1] + _rN2*ρb*(-ΔHr) + 4*U/d*Tr #temperatura

    #condições de contorno em z=L
    res[N] = 3*CN2[N] - 4*CN2[N-1] + CN2[N-2] #vazão molar
    res[2*N+1] = 3*T[N] - 4*T[N-1] + T[N-2]  #temperatura

    """     #deixando tudo em escala
    res[0] /= abs(C0N2)
    res[N+1] /= abs(Tin)

    for i in range(1, N):
        res[i] /= abs(C0N2)
        res[N+1 + i] /= abs(Tin) """

    return res

# resolvendo
resultado = fsolve(fobj, estimativa)

# printando
CN2 = resultado[:N+1]
T = resultado[N+1:]

print("\n")
print(f"C0N2 = {C0[0]:.1f} mol/m³")
print("Concentração molar (mol/m³):", CN2.round(1))
print("Conversão:", (100*(C0N2-CN2)/C0N2).round(1))
print("Temperatura (ºC):", (T-273.15).round(0))
print(f"\n Resíduo:{np.linalg.norm(fobj(resultado)):.3g}")
print((CN2-CN2_est).round(1))
