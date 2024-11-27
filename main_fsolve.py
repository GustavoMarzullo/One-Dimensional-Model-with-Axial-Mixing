import numpy as np
from scipy.optimize import fsolve, root
from Reator.conversao import rN2
import matplotlib.pyplot as plt

#vazão de entrada
FT = 6500 #mol/s
F0N2=0.21825*FT
F0H2=0.65475*FT
F0NH3=0.05*FT
F0 =np.array([F0N2, F0H2, F0NH3])

Tin = 380 + 273.15 #K
Tr = 25 + 273.15 #K
Pin = 155 #atm

ε = 0.4
Ac = 7 #m²
U = 30 #W/(m².K)
d = 2*np.sqrt(Ac/np.pi) #m
ρb = 1 #kg/m³

Dea = 1e-4 #m²/s
λea = 4e-3 #m²/s
Cp = 5000 #J/(kg.K)
ΔHr = -46e3 #J/mol_N2

M = 9e-3 #kg/mol
ρG = 23 #kg/m³
Q = FT*M/ρG #m³/s
C0 = F0/Q
C0N2, C0H2, C0NH3 = C0
v = Q/Ac #m³/s -> m/s
us = v*ε #velocidade superficial do gás
print(Q, v, us)

L0 = 0 #valor inicial de comprimento
LF = 5 #valor final de comprimento
N = 100 #número de espaços
h = (LF-L0)/N #step size
L_eval = np.linspace(L0, LF, N+1)

A = (ε*Dea)/(h**2)
B = (-2*ε*Dea)/(h**2) - (3*us)/(2*h)
C = (ε*Dea)/(h**2) + (2*us)/h
A1 = -us/(2*h)
D =  (3*ε*Dea)/(2*h) + us
E = (-2*ε*Dea)/(h)
F = (ε*Dea)/(2*h)
G = (λea)/(h**2) - (ρG*us*Cp)/(2*h)
H = (-2*λea)/(h**2) - 4*U/d
I = (λea)/(h**2) + (ρG*us*Cp)/(2*h)
J = (3*λea)/(2*h) + (ρG*us*Cp)
K = (-2*λea)/(h)
L =(λea)/(2*h)

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
        if i == 1:
            res[i] =  A*CN2[i+1] + ((-2*ε*Dea)/(h**2) - us/h)*CN2[i] + ((ε*Dea)/(h**2) + us/h)*CN2[i-1] - _rN2*ρb #vazão molar
        else:
            res[i]=  A*CN2[i+1] + B*CN2[i] + C*CN2[i-1] + A1*CN2[i-2] - _rN2*ρb #vazão molar

        res[N+1+i] = G*T[i+1] + H*T[i] + I*T[i-1] + _rN2*ρb*(-ΔHr) + 4*U/d*Tr #temperatura

    #condições de contorno em z=L
    res[N] = (3*CN2[N] - 4*CN2[N-1] + CN2[N-2]) #vazão molar
    res[2*N+1] = (3*T[N] - 4*T[N-1] + T[N-2])  #temperatura

     #deixando tudo em escala
    res[0] /= abs(C0N2)
    res[N+1] /= abs(Tin)

    for i in range(1, N):
        res[i] /= abs(C0N2)
        res[N+1 + i] /= abs(Tin)

    return res

#estimativas iniciais
CN2_est = np.linspace(1.0*C0N2, 0.75*C0N2, N + 1)
T_est = np.linspace(Tin, Tin+20, N + 1)
estimativa = np.concatenate([CN2_est, T_est])

# resolvendo
resultado = root(fobj, estimativa, method='hybr').x

# printando
CN2 = resultado[:N+1]
T = resultado[N+1:]

print("\n")
print(f"C0N2 = {C0[0]:.1f} mol/m³")
print("Concentração molar (mol/m³):", CN2.round(1))
print("Conversão:", (100*(C0N2-CN2)/C0N2).round(1))
print("Temperatura (ºC):", (T-273.15).round(0))
residuo = np.linalg.norm(fobj(resultado))
print(f"\n Resíduo:{residuo:.3g}")
print((CN2-CN2_est).round(1))

if residuo < 1e-3:
    # Plots pressure and temperature
    fig, ax = plt.subplots()
    ax.plot(L_eval, CN2, color='black', linewidth=0.8)
    ax1 = ax.twinx()
    ax1.plot(L_eval, T-273.15, color='red', linewidth=0.8)
    ax.set_xlabel("Comprimento (m)")
    ax.set_ylabel("Concentração (mol/m³)")
    ax1.set_ylabel("Temperatura (°C)", color='red')
    ax1.tick_params(colors="red")
    print(CN2.round(0))
    print(T.round(0)-273)
    plt.show()
