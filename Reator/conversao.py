import numpy as np
import sys
sys.path.append('Reator')

def coeficiente_atividade(
        P: float,
        T: float) -> np.ndarray:
    """
    Função para calcular a taxa de reação.
    Fonte: Bischoff (1979)

    Parameters
    ----------
    P
        Pressão [atm].
    T
        Temperatura [K].
    
    Returns
    -------
    Taxa de reação. [kmol/(kg_cat.h)]
    """
    γN2 = 0.93431737 + 0.310180e-3*T + 0.295896e-3*P - 0.270727e-6*T**2 + 0.4775207e-6*P**2
    γH2 = np.exp(np.exp(-3.8402*T**0.125 +0.541)*P - np.exp(-0.1263*T**0.5 -15.98)*P**2 + 300*(np.exp(-0.011901*T-5.491)*(np.exp(-P/300)-1)))
    γNH3 = 0.1438996 +  0.2028538e-2*T - 0.4487672e-3*P - 0.1142945e-5*T**2 + 0.2761216e-6*P**2


    return np.array([γN2, γH2, γNH3])

def constante_velocidade(T: float) -> float:
    """
    Função para calcular a constante de velocidade da formação da amônia.
    Fonte: Suhan et. al (2018)

        Parameters
    ----------
    T
        Temperatura [K].
    
    Returns
    -------
    Constante de velocidade da formação da amônia [adimensional]
    """
    E = 170560.76 #kJ/kmol
    R = 8.314 #kJ/(kmol.K)
    k0 = 8.849e14
    k = k0*np.exp(-E/(R*T))
    return k

def constante_equilibrio_amonia(T:float) -> float:
    """
    Função para calcular a constante de equilibrio da amônia.
    Fonte: Dyson (1968)

        Parameters
    ----------
    T
        Temperatura [K].
    
    Returns
    -------
    Constante de equilíbrio da amônia [adimensional]
    """
    Ka = 10**(-2.691122*np.log10(T) - 5.519265e-5*T + 1.848863e-7*T**2 + 2001.6/T + 2.6899)
    return Ka

###constantes do fator de efetividade do nitrogênio
B0 = [-17.539096, -8.2125534, -4.6757259]
B1 = [0.07697849, 0.03774149, 0.02354872]
B2 = [6.900548, 6.190112, 4.687353]
B3 = [-1.082790e-4, -5.354571e-5, -3.463308e-5]
B4 = [-26.424699, -20.86963, -11.28031]
B5 = [4.927648e-8, 2.379142e-8, 1.540881e-8]
B6 = [38.93727,	27.88403, 10.46627]
Pressoes_B = [150, 225, 300]
###

def constantes_fator_efetividade_N2(P:float) -> np.ndarray:
    """
    Função para calcular as constantes do fator de efetividade do nitrogênio.
    Fonte: Dyson (1968)

        Parameters
    ----------
    P
        Pressão [atm].
    
    Returns
    -------
    Constantes do fator de efetividade do nitrogênio [adimensional]
    """
    b0 = np.interp(P, Pressoes_B, B0)
    b1 = np.interp(P, Pressoes_B, B1)
    b2 = np.interp(P, Pressoes_B, B2)
    b3 = np.interp(P, Pressoes_B, B3)
    b4 = np.interp(P, Pressoes_B, B4)
    b5 = np.interp(P, Pressoes_B, B5)
    b6 = np.interp(P, Pressoes_B, B6)

    B = np.array([b0, b1, b2, b3, b4, b5, b6])

    return B

def fator_efetividade_N2(
        P: float,
        T: float,
        X: float) -> float:
    """
    Função para calcular o fator de efetividade do nitrogênio.
    Fonte: Dyson (1968)

    Parameters
    ----------
    P
        Pressão [atm].
    T
        Temperatura [K].
    X
        Conversão de amônia [adimensional].
    
    Returns
    -------
    Fator de efetividade do nitrogênio. [adimensional]
    """
    Bcalc = constantes_fator_efetividade_N2(P)
    η =  Bcalc[0] + Bcalc[1]*T + Bcalc[2]*X + Bcalc[3]*T**2 + Bcalc[4]*X**2 + Bcalc[5]*T**3 + Bcalc[6]*X**3

    return η

def rN2(
        F:float,
        T: float,
        P:float)-> float:    
    """
    Função para calcular a taxa de reação da reação de conversão de nitrogênio.
    Fonte: Jorqueira et. al (2018)

    Parameters
    ----------
    F
        Vazão molar de nitrogênio, hidrogênio e amônia [mol/s].
    T
        Temperatura [K].
    P
        Pressão na entrada do reator [atm].
    Returns
    -------
    Taxa de reação de amônia do reator [mol/(m³.s)];
    """
    #calculando as concentrações
    Y = F/sum(F) #fração molar

    Ka = constante_equilibrio_amonia(T)
    k = constante_velocidade(T)
    aN2, aH2, aNH3 = Y *coeficiente_atividade(P, T) * P
    rNH3 = 2*k*(Ka**2 * aN2*(aH2**3/aNH3**2)**0.5 - (aNH3**2/aH2**3)**0.5) #kmol/(m³.h)
    rNH3 = rNH3/3.6 #kmol/(m³.h) -> mol/(m³.s)
    rN2 = -2*rNH3
    #assert rNH3 >=0
    
    return rN2





