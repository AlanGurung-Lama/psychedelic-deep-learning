# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:14:01 2024

@author: Alan
 * This program is based on a java based script 
 * PinkNoise.java  -  a pink noise generator
 *
 * Copyright (c) 2008, Sampo Niskanen <sampo.niskanen@iki.fi>
 * All rights reserved.
 * Source:  http://www.iki.fi/sampo.niskanen/PinkNoise/
 *
 * Distrubuted under the BSD license:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *  - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 
 *  - Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 *
 *  - Neither the name of the copyright owners nor contributors may be
 * used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn




class PinkNoise:
    def __init__(self, beta=1.0, poles=20, random_state=None):
        if beta < 0 or beta > 2:
            raise ValueError("Invalid pink noise beta = {}".format(beta))
        
        self.beta = beta
        self.poles = poles
        self.multipliers = np.zeros(poles)
        self.values = np.zeros(poles)
        
        
        if random_state is None:
            self.random = np.random.RandomState()
        else:
            self.random = np.random.RandomState(random_state)
        
        a = 1.0
        for i in range(poles):
            a = (i - beta / 2.0) * a / (i + 1.0)
            self.multipliers[i] = a
        
        # Fill the history with random values
        for _ in range(5 * poles):
            self.next_value()

    def next_value(self):
        x = self.random.randn()
        for i in range(self.poles):
            x -= self.multipliers[i] * self.values[i]
        
        self.values[1:] = self.values[:-1]
        self.values[0] = x
        
        return x

def generate_pink_noise(beta=1.0, alpha=1.0, n_samples=1000, poles=20, random_state=None, scaling_factor=0.005):
    """
    Generate pink noise with given parameters.
    
    Parameters:
    - beta: float, the exponent of the pink noise, 1/f^beta.
    - alpha: float, factor to scale the amplitude of the noise.
    - n_samples: int, number of samples to generate.
    - poles: int, number of poles to use for the IIR filter.
    - random_state: int or None, seed for the random number generator.
    
    Returns:
    - noise: np.ndarray, generated pink noise time-series data.
    """
    pink_noise_generator = PinkNoise(beta=beta, poles=poles, random_state=random_state)
    noise = np.array([pink_noise_generator.next_value() for _ in range(math.ceil(n_samples+100))])
    noise = noise[99:]
    noise = noise / np.std(noise)
    # Convert alpha to a float if it is an instance of Parameter
    if isinstance(alpha, nn.Parameter):
        alpha = alpha.item()  # Convert torch tensor to float
    
    noise *= alpha
    return noise*scaling_factor

# #%%
# from fooof import FOOOF
# from scipy.signal import welch
# # Usage example
# if __name__ == "__main__":
#     beta = 1.0
#     alpha = 1.0
#     n_samples = 1000
    
#     noise = generate_pink_noise(beta=beta, alpha=alpha, n_samples=n_samples)
    
#     # Sampling parameters
#     Fs = 10000  # Sampling frequency
    
   
#     # Plot the results
#     plt.figure(figsize=(12, 6))
    
#     # Plot the original signal
#     plt.subplot(2, 1, 1)
#     plt.plot(noise)
#     plt.title(rf"New model's noise time series with $\beta$  = {beta}")
#     plt.xlabel('Sample Index')
#     plt.ylabel('Amplitude')
#     plt.show()
    
#     # Compute the Power Spectral Density (PSD) using Welch's method
#     f, Pxx = welch(noise, Fs, nperseg=n_samples)
    
#     # Avoid dividing by zero by starting from first non-zero frequency
#     non_zero_indices = f != 0
  
#     # Initialize a FOOOF object
#     fm = FOOOF()

#     # Set the frequency range to fit the model
#     freq_res = f[1]-f[0]
#     freq_range = [2*freq_res, Fs/2]

#     # Report: fit the model, print the resulting parameters, and plot the reconstruction
#     fm.fit(f, Pxx, freq_range)
    
#     # Check the aperiodic parameters
#     offset, exponent = fm.aperiodic_params_
       
#     # # Perform linear regression
#     # slope, intercept, r_value, p_value, std_err = linregress(np.log10(f[non_zero_indices]), 10 * np.log10(Pxx[non_zero_indices]))
    
    
#     # Calculate the line of best fit
#     fit_line = -10*exponent* np.log10(f[non_zero_indices]) + 10*offset
    
#     # # Calculate simulated beta
#     # sim_beta = slope/(-10)
#     print(f"Simulated beta = {exponent}")
    
    
#     # Calculate theoretical fit
#     theo_line = -10*beta *np.log10(f[non_zero_indices]) + 10*offset
#     diff = fit_line[0]-theo_line[0]
#     theo_line += diff
    
#     # Plot the frequency plot with power and logarithmic frequency axis   
#     plt.subplot(2, 1, 2)
#     # Plot the PSD
#     plt.semilogx(f[non_zero_indices], 10 * np.log10(Pxx[non_zero_indices]),'o', label='PSD')
    
#     # Plot the line of best fit
#     plt.semilogx(f[non_zero_indices], fit_line, label='PSD line of best fit', linestyle='--', color='red')
    
#     # Plot theoretical line
#     plt.semilogx(f[non_zero_indices], theo_line, label=r'1/f$^\beta$', linestyle='--', color='green')
    
#     plt.title("New model's power spectral density with the theoretical spectrum")
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Power/Frequency [dB/Hz]')
#     plt.grid(True, which="both", ls="--")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'New_time_series_and_PSD_beta={beta}.png')
#     plt.show()
    