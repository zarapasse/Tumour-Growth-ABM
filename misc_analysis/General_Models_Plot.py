#Want to plot the 4 common general models used in the Introduction section of the paper
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,50,500)
V_0 = 0.1

#Mendelsohn
r_m = 0.1
C_m = 0.5 
V_mendelsohn = (1/3 * (r_m*t +C_m))**3

#Gompertz
r_g = 0.6
rho_g = 0.15
V_gompertz = V_0 * np.exp(r_g/rho_g * (1 - np.exp(-rho_g*t)))

#Logistic 
r_l = 0.5
K_l = 5
V_logistic = (K_l*V_0*np.exp(r_l*t)) / (K_l + V_0*(np.exp(r_l*t)-1))

#Bertalanffy Model Parameters
alpha_b = 0.45
beta_b = 0.25
V_bertalanffy = ( alpha_b/beta_b - (alpha_b / beta_b - V_0**(1/3))*np.exp(-beta_b*t/3))**3

# Determine common y-axis limit
V_max = max(V_mendelsohn.max(), V_gompertz.max(), V_logistic.max(), V_bertalanffy.max())

# Create 4 subplots
fig, axs = plt.subplots(1, 4, figsize=(20,5))

models = [V_mendelsohn, V_gompertz, V_logistic, V_bertalanffy]
titles = ['Mendelsohn', 'Gompertz', 'Logistic', 'Bertalanffy']
colors = ['blue', 'orange', 'green', 'red']

for ax, V, title, color in zip(axs, models, titles, colors):
    ax.plot(t, V, color=color)
    ax.set_title(title)
    ax.set_xlabel('t')
    ax.set_ylabel('V(t)')
    ax.set_ylim(0, V_max*1.05)  # slightly larger to avoid touching top
    ax.grid(True)

plt.tight_layout()
plt.show()