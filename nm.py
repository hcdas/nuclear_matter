# author H. C. Das 
# 10 january 2022

import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib import rc
from pylab import *
from matplotlib.ticker import (MultipleLocator)

rc('text', usetex=True)
rc('font', family='serif', weight='bold')
rc('axes', linewidth=1.5)

plt.rc('axes',linewidth=1.5)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=16)

fontsize = 14  # for tick label

#==========================================#

hc = 197.33      # hc = 197.33 MeV fm 

gam_snm = 4      # for SNM

gam_pnm = 2      # for pnm

# Mass of nucleons      
mn = 4.758   # fm^-1    

# spacing
dt = 0.01    

#++++++++++++++++++++++++++++++++++++++++++++#
# for symmetric nuclear matter, gamma = 4
def fun_snm(m_snm,kf):

    ef = np.sqrt(m_snm**2+kf**2)   # fm^-1

    m_eff_snm = m_snm - mn + ((267.1/mn**2) \
              *(gam_snm*m_snm/(4.*pi**2)))*(kf*ef-(m_snm**2*np.log((kf+ef)/m_snm))) # fm^-1

    return m_eff_snm

# pure for neutron matter, gamma = 2
def fun_pnm(m_pnm,kf):          

    ef = np.sqrt(m_pnm**2+kf**2)  # fm^-1

    m_eff_pnm = m_pnm - mn + ((267.1/mn**2) \
              *(gam_pnm*m_pnm/(4.*pi**2)))*(kf*ef-(m_pnm**2*np.log((kf+ef)/m_pnm))) # fm^-1

    return m_eff_pnm

# for symmetric nuclear matter, gamma = 4
def energy_snm(m_snm,kf):

    rhob = gam_snm*kf**3/(6*pi**2)      # fm^-3

    ef = np.sqrt(m_snm**2+kf**2)        # fm^-1

    e1 = 195.9*rhob**2/(2.0*mn**2)      # fm^-4

    e2 = mn**2*(mn-m_snm)**2/(2*267.1)  # fm^-4

    e3 = gam_snm/(16.0*pi**2)*(kf*ef*(2*kf**2+m_snm**2)-m_snm**4*np.log((kf+ef)/m_snm)) # fm^-4

    e_snm = e1+e2+e3                    # fm^-4

    return e_snm

# for pure neutron matter, gamma = 2
def energy_pnm(m_pnm,kf):

    rhob = gam_pnm*kf**3/(6*pi**2)      # fm^-3

    ef = np.sqrt(m_pnm**2+kf**2)        # fm^-1

    e1 = 195.9*rhob**2/(2.0*mn**2)      # fm^-4

    e2 = mn**2*(mn-m_pnm)**2/(2*267.1)  # fm^-4

    e3 = gam_pnm/(16.0*pi**2)*(kf*ef*(2*kf**2+m_pnm**2)-m_pnm**4*np.log((kf+ef)/m_pnm)) # fm^-4

    e_pnm = e1+e2+e3                    # fm^-4

    return e_pnm

# for symmetric nuclear matter, gamma = 4
def pressure_snm(m_snm,kf): 

    rhob = gam_snm*kf**3/(6*pi**2)         # fm^-3

    ef = (m_snm**2+kf**2)**0.5             # fm^-1

    p1 = 0.5*195.9*rhob**2/mn**2           # fm^-4

    p2 = 0.5*mn**2*(mn-m_snm)**2/267.1     # fm^-4

    p3 = (gam_snm/(48*pi**2))*(kf*ef*(2*kf**2-3*m_snm**2) + 3*m_snm**4*np.log((kf+ef)/m_snm))  # fm^-4

    p_snm = p1-p2+p3

    return p_snm

# for pure neutron matter, gamma = 2
def pressure_pnm(m_pnm,kf):  

    rhob = gam_pnm*kf**3/(6*pi**2)         # fm^-3

    ef = (m_pnm**2+kf**2)**0.5             # fm^-1

    p1 = 0.5*195.9*rhob**2/mn**2           # fm^-4
    
    p2 = 0.5*mn**2*(mn-m_pnm)**2/267.1     # fm^-4
    
    p3 = (gam_pnm/(48*pi**2))*(kf*ef*(2*kf**2-3*m_pnm**2) + 3*m_pnm**4*np.log((kf+ef)/m_pnm))  # fm^-4

    p_pnm = p1-p2+p3

    return p_pnm

# for SNM
mom_snm = []
rho_snm = []
effm_snm = []
ener_snm = []
press_snm = []
be_snm = []

for i in range(1,270):
      kf = i*dt
      mom_snm.append(kf)

      # for SNM
      r_snm = gam_snm*kf**3/(6*pi**2)
      rho_snm.append(r_snm)
      sol_s = optimize.root(fun_snm, ([4.5,0.01]), method='lm', args=(kf))
      effm_snm.append(sol_s.x[0]/4.758)
      ener_snm.append(energy_snm(sol_s.x[0],kf))
      press_snm.append(pressure_snm(sol_s.x[0],kf))
      be_snm.append(((energy_snm(sol_s.x[0],kf)/r_snm)*hc)-939.)

# for PNM
mom_pnm = []
rho_pnm = []
effm_pnm = []
ener_pnm = []
press_pnm = []
be_pnm = []

for i in range(1,270):
      kf = i*dt
      # for PNM
      mom_pnm.append(kf)
      r_pnm = gam_pnm*kf**3/(6*pi**2)
      rho_pnm.append(r_pnm)
      sol_p = optimize.root(fun_pnm, ([4.5,0.01]), method='lm', args=(kf))
      effm_pnm.append(sol_p.x[0]/4.758)
      ener_pnm.append(energy_pnm(sol_p.x[0],kf))
      press_pnm.append(pressure_pnm(sol_p.x[0],kf))
      be_pnm.append(((energy_pnm(sol_p.x[0],kf)/r_pnm)*hc)-939.)

OUT_snm = np.c_[mom_snm, rho_snm, effm_snm, ener_snm, press_snm, be_snm];

# Output to file snm.dat
np.savetxt('snm.dat', OUT_snm, fmt='%1.8e')

OUT_pnm = np.c_[mom_pnm, rho_snm, effm_snm, ener_snm, press_snm, be_snm];

# Output to file pnm.dat
np.savetxt('pnm.dat', OUT_pnm, fmt='%1.8e')

print(min(be_snm))

fig = plt.figure(figsize=(6,7))
gs = fig.add_gridspec(2, 1, hspace=0, wspace=0)
ax = gs.subplots(sharex=True, sharey=False)

ax[0].plot(mom_snm,effm_snm,label='SNM',lw='2')
ax[0].plot(mom_pnm,effm_pnm,label='PNM',lw='2')

ax[0].set(ylabel=r'$M^*/M$')
ax[0].set(xlabel=r'$k_f$ (fm$^{-1}$)')

ax[0].set_ylim([0, 1.0])
ax[0].set_xlim([0.01, 2.8])

ax[0].legend(fontsize='large')
ax[0].vlines(1.42, -20, 100, color='red', linestyle='--')

ax[1].plot(mom_snm,be_snm,label='SNM',lw='2')
ax[1].plot(mom_pnm,be_pnm,label='PNM',lw='2')

ax[1].set(ylabel=r'BE (MeV)')
ax[1].set(xlabel=r'$k_f$ (fm$^{-1}$)')

ax[1].set_ylim([-20, 119])
ax[1].set_xlim([0.01, 2.8])

#ax[1].legend(fontsize='large')

ax[1].hlines(0.0, 0.01, 3, color='red', linestyle='--')
ax[1].vlines(1.42, -20, 120, color='red', linestyle='--')

ax[0].xaxis.set_major_locator(MultipleLocator(0.4))
ax[0].yaxis.set_major_locator(MultipleLocator(0.2))

ax[0].xaxis.set_minor_locator(MultipleLocator(0.2))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.1))

ax[1].xaxis.set_major_locator(MultipleLocator(0.4))
ax[1].xaxis.set_minor_locator(MultipleLocator(0.2))


ax[1].yaxis.set_major_locator(MultipleLocator(20))
ax[1].yaxis.set_minor_locator(MultipleLocator(10))


for tick in ax[0].yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')

for tick in ax[0].xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')

for tick in ax[1].xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')

for tick in ax[1].yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')

ax[0].tick_params(axis='x', which="major", direction="in",length=8, width=1,
              top=True, right=True)
ax[0].tick_params(axis='x', which="minor", direction="in",length=4, width=1,
               top=True, right=True)

ax[0].tick_params(axis='y', which="major", direction="in",length=8, width=1,
               top=True, right=True)
ax[0].tick_params(axis='y', which="minor", direction="in",length=4, width=1,
               top=True, right=True)

ax[1].tick_params(axis='x', which="major", direction="in",length=8, width=1,
              top=True, right=True)
ax[1].tick_params(axis='x', which="minor", direction="in",length=4, width=1,
               top=True, right=True)

ax[1].tick_params(axis='y', which="major", direction="in",length=8, width=1,
               top=True, right=True)
ax[1].tick_params(axis='y', which="minor", direction="in",length=4, width=1,
               top=True, right=True)

plt.show()
plt.close()
plt.tight_layout()

fig.savefig('nm.png', format='png', dpi=300)
