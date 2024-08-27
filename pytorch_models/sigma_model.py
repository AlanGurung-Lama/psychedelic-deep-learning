# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:06:25 2024

@author: Alan
"""
import numpy as np
import torch
from scipy.integrate import odeint
from Pink_Noise_Java import generate_pink_noise
import matplotlib.pyplot as plt

def explicit_euler_w_pink_noise(x0, pink_noise, t, params):
    """
    Solves ODEs using the explicit Euler method.

    Parameters:
    - ode_func: Function that returns dy/dt given y, t, and parameters
    - y0: Initial condition
    - t: Array of time points
    - params: Parameters for the ODE function

    Returns:
    - y: Array of solution values
    """
  
    
    # Neuronal models
    x_E = torch.zeros(len(t))
    x_I = torch.zeros(len(t))
    
    sigma = params[0]
    mu = params[1]
    lamda = params[2]
    c = params[3]
    k = params[4]
    
    # Neurovascular coupling
    a = torch.zeros(len(t))
    f = torch.zeros(len(t))
    a[0] = 0
    f[0] = 1
    
    phi = params[5]
    cphi = params[6]
    chi = params[7]
    
    # Hemodynamic model
    def E(f):
        output = 1 - ( 1 - E_0) ** (1/f)
        return output

    v = torch.zeros(len(t))
    q = torch.zeros(len(t))
    f_out = torch.zeros(len(t))
    v[0] = 1
    q[0] = 1
    f_out[0] = 1
    
    t_MTT = params[8]
    tau = params[9]
    alpha = params[10]
    E_0 = params[11]
    
    V_0 = params[12]
    k_1 = params[13]
    k_2 = params[14]
    k_3 = params[15]
    
    #Physical BOLD signal
    y = torch.zeros(len(t))
    
    for i in range(0,len(t) - 1):
        x_E[i+1] = x_E[i] + h * (- sigma * x_E[i] - mu * x_I[i] + c * u[i] )
        x_I[i+1] = x_I[i] + h * (lamda * (x_E[i] - x_I[i]))
        a[i+1] = a[i] + h * (-cphi * a[i] + x_E[i])
        f[i+1] = f[i] + h * (phi * a[i] - chi * (f[i] - 1))
        v[i+1] = v[i] + h * ((f[i] - f_out[i]) / t_MTT)
        q[i+1] = q[i] + h * (( f[i] * (E(f[i]) / E_0) - f_out[i] * (q[i] / v[i]) ) / t_MTT )
        f_out[i+1] = ((t_MTT * v[i] ** (1 / alpha) + tau * f[i]) / (t_MTT + tau))

    y = torch.multiply(V_0, k_1 * ( 1 - q ) + k_2 * (1 - q / v ) + k_3 * (1 - v))   
  
    # plt.figure()
    # plt.ylabel('Neuronal activity')
    # plt.xlabel('Time')
    # plt.plot(t,x_E,label='Excitatory')
    # plt.plot(t,x_I,label='Inhibitory')
    # plt.legend()

    # plt.figure()
    # plt.ylabel('Vasoactive signal a(t)')
    # plt.xlabel('Time')
    # plt.plot(t,a)
    
    # plt.figure()
    # plt.ylabel('Blood flow f(t)')
    # plt.xlabel('Time')
    # plt.plot(t,f)
    
    # plt.figure()
    # plt.ylabel('Blood volume v(t)')
    # plt.xlabel('Time')
    # plt.plot(t,v)
    
    # plt.figure()
    # plt.ylabel('Deoxyghemoglobin content q(t)')
    # plt.xlabel('Time')
    # plt.plot(t,q)
    
    # plt.figure()
    # plt.ylabel('Blood outflow f_out(t) ')
    # plt.xlabel('Time')
    # plt.plot(t,f_out,label='Blood outflow')
     
    plt.figure()
    plt.title('BOLD signal comparison')
    plt.plot(t,y,label='Simulated')
    print(y)
    return y

# Parameters
h = torch.tensor(0.01)  # step size
time = torch.arange(0, 217+h, h)  # Time vector

# Generate exogenous input u(t)
u = generate_pink_noise(beta=1,alpha=1, n_samples=len(time)-1)
u= torch.from_numpy(u)
x0 = [torch.tensor(0), 
      torch.tensor(0),
      torch.tensor(0),
      torch.tensor(1),
      torch.tensor(1),
      torch.tensor(1),
      torch.tensor(1)]

params = [torch.tensor(0.5),
          torch.tensor(0.4),
          torch.tensor(0.2),
          torch.tensor(1),
          torch.tensor(0),
          torch.tensor(0.6),
          torch.tensor(1.5),
          torch.tensor(0.6),
          torch.tensor(2),
          torch.tensor(4),
          torch.tensor(0.32),
          torch.tensor(0.4),
          torch.tensor(4),
          torch.tensor(4.85212),
          torch.tensor(0.3315816),
          torch.tensor(0.7807)]


# Solve the ODE using the explicit Euler method
solution = explicit_euler_w_pink_noise(x0, u, time, params)
