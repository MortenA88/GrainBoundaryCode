# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import kwant
import numpy as np
import tinyarray as ta
import scipy.sparse.linalg as sla
import pandas as pd
#from Postprocessor import *
import time

name = 'foldername'

# Pauli matrices
sigma_0 = ta.array([[1,0],[0,1]])
sigma_1 = ta.array([[0,1],[1,0]])
sigma_2 = ta.array([[0,-1j],[1j,0]])
sigma_3 = ta.array([[1,0],[0,-1]])


# Hopping parameter
t = 1.

# Chemical potential
mu = 5.85

# Fermi momentum times lattice constant
kF = np.sqrt(mu/t)

# helical p-wave SC
dp = 0.75*t/kF
def Msc(angle):

    Mscx =  0.5*1j*dp*np.cos(angle)*np.kron(sigma_2,sigma_0) + 0.5*1j*dp*np.sin(angle)*np.kron(sigma_1,sigma_3)
    Mscy = -0.5*1j*dp*np.sin(angle)*np.kron(sigma_2,sigma_0) + 0.5*1j*dp*np.cos(angle)*np.kron(sigma_1,sigma_3)

    return [Mscx,Mscy]

## Spin-orbit coupling
#asoc = 0.
#Msocx =  0.5*1j*asoc*np.kron(sigma_3,sigma_2)
#Msocy = -0.5*1j*asoc*np.kron(sigma_0,sigma_1)

# Exchange field
hq = 0.1*kF*0.
h = 0.28
ha = 90.
hx = h*np.cos(ha*np.pi/180)
hy = h*np.sin(ha*np.pi/180)
hz = 0.0
ds = 0.

def Mh():
    return hx*np.kron(sigma_3,sigma_1) + hy*np.kron(sigma_0,sigma_2) + hz*np.kron(sigma_3,sigma_3) + ds*np.kron(sigma_2,sigma_2)



# Distance between matching lattice points
d = 12

# Shape accuracy
ge = 1.e-3;

# Lattice misalignment
a = np.arctan(1./d)
a0 = a

# Number of points at the bottom of the lattice
Nb = 31

# Number of points in the width of the lattice, in increments of d
nw =69

Nw = nw*d

# Dimensions of lattice
Lt = Nb - 1
Lw = Nw - 1

# Center point
xc = Lt*np.cos(a) - 2.*Lw*np.sin(a)
yc = Lt*np.sin(a)
xcom = Lw*np.sin(a)

# Primitive lattice vectors
pa = [(np.cos(a),np.sin(a)),(-np.sin(a),np.cos(a))]
pb = [(np.cos(-a),np.sin(-a)),(-np.sin(-a),np.cos(-a))]

# Offset between sublattices (slip)
noff = 5
offset = np.array((0.,noff/pb[1][1]))

lata = kwant.lattice.Monatomic(pa,name='a')
latb = kwant.lattice.Monatomic(pb,offset,name='b')

def rotate(pos,angle):
    x,y = pos
    xp = x*np.cos(angle) + y*np.sin(angle)
    yp = -x*np.sin(angle) + y*np.cos(angle)
    return xp,yp


# Shape functions
def left(pos):
    xp,yp = rotate(pos,a) 
    return xp + Lt > ge and pos[0] <= ge and yp >= noff-ge and yp - Nw <= ge


def right(pos):
    xp,yp = rotate(pos,-a)
    return pos[0] >= -ge and xp - Lt <= ge and yp >= noff-ge and yp - Nw <= ge



syst = kwant.Builder()

def onsiteLeft(site):
    return (4*t-mu) * np.kron(sigma_3,sigma_0) + Mh()

def onsiteRight(site):
    return (4*t-mu) * np.kron(sigma_3,sigma_0) + Mh()

# Local site potential
syst[ lata.shape(left,(-2,noff+1))] = onsiteLeft
syst[ latb.shape(right,(2,noff+1))] = onsiteRight


# Find the connecting edges
Sites = list(syst.sites())
aSites = [s for s in Sites if s[0].name == 'a']
bSites = [s for s in Sites if s[0].name == 'b']


# Identify each row in the rotate lattice
ppa = np.stack([rotate(s.pos,a) for s in aSites])
ppb = np.stack([rotate(s.pos,-a) for s in bSites])
pay0 = np.round(ppa[:,1])
pby0 = np.round(ppb[:,1])
pay = np.unique(pay0)
pby = np.unique(pby0)
EdgeNode_a = []
EdgeNode_b = []
pbc_a = []
pbc_b = []


# Find highest (lowest) x position of site in each row for lattice a (b),
for y in pay:
    xa = np.max([p[0] for p in ppa if np.round(p[1]) == y])
    xa0 = np.min([p[0] for p in ppa if np.round(p[1]) == y]) 
    dca = np.argwhere((ppa[:,0] == xa) & (pay0 == y))
    dc0 = np.argwhere((ppa[:,0] == xa0) & (pay0 == y)) 
    EdgeNode_a.append(aSites[int(dca)])
    pbc_a.append(aSites[int(dc0)])
    
for y in pby:
    xb = np.min([p[0] for p in ppb if np.round(p[1]) == y])
    xb0 = np.max([p[0] for p in ppb if np.round(p[1]) == y])
    
    dcb = np.argwhere((ppb[:,0] == xb) & (pby0 == y))
    dc0 = np.argwhere((ppb[:,0] == xb0) & (pby0 == y))
    EdgeNode_b.append(bSites[int(dcb)])
    pbc_b.append(bSites[int(dc0)])


# Find nodes located exactly at the center
CenterNode = []
for s in Sites:
    if np.abs(s.pos[0]) <= 1.e-3:
        CenterNode.append(s)
        
pC = [p.pos for p in CenterNode]
ConNode = []
for c in CenterNode:
    
    di = np.argmin([np.sqrt( 2.*(c.pos[0] - s.pos[0])**2 + (c.pos[1] - s.pos[1])**2) for s in bSites])
    ConNode.append(bSites[di])
    
    


#a = 0
xb = np.min([en.pos[0] for en in EdgeNode_b])
xa = np.max([en.pos[0] for en in EdgeNode_a])


# Hopping contributions
syst[kwant.builder.HoppingKind((1, 0), lata, lata)] =  -t*np.kron(sigma_3,sigma_0) + Msc(a)[0] 
syst[kwant.builder.HoppingKind((0, 1), lata, lata)] =  -t*np.kron(sigma_3,sigma_0) + Msc(a)[1]
syst[kwant.builder.HoppingKind((1, 0), latb, latb)] =  -t*np.kron(sigma_3,sigma_0) + Msc(-a)[0]
syst[kwant.builder.HoppingKind((0, 1), latb, latb)] =  -t*np.kron(sigma_3,sigma_0) + Msc(-a)[1]


for j in range(len(EdgeNode_a)):
    
    if np.abs(EdgeNode_a[j].pos[0]) <= 1.e-3:
        # Jumping happens along the x-diraction of lattice b for edge dislocations on lattice a
        syst[(EdgeNode_a[j],EdgeNode_b[j])] = -t*np.kron(sigma_3,sigma_0) + Msc(-a)[0]
    elif np.abs(EdgeNode_b[j].pos[0]) <= 1.e-3:
        # Jumping happens along the x-direction of lattice a for edge dislocations on lattice b
        syst[(EdgeNode_a[j],EdgeNode_b[j])] = -t*np.kron(sigma_3,sigma_0) + Msc(a)[0]
    else:
        # Jumping happens horizontally for off-center sites
        syst[(EdgeNode_a[j],EdgeNode_b[j])] = -t*np.kron(sigma_3,sigma_0) + Msc(0)[0]
    
# Solve system
syst = syst.finalized()
ham_mat = syst.hamiltonian_submatrix(sparse=True)
e,v = sla.eigsh(ham_mat.tocsc(), k=50, which='LM', sigma=0., return_eigenvectors=True)
eind = np.argsort(np.abs(e))
e = e[eind]
v = v[:,eind]

# A selection of outputs
rho = LDOS(e,v,0.)
px = [s.pos[0] for s in syst.sites]
py = [s.pos[1] for s in syst.sites]
vbot = v[:,:16]
rvb = np.real(vbot.transpose().flatten())
ivb = np.imag(vbot.transpose().flatten())

ldosoutput = pd.DataFrame({'x':px, 'y':py, 'ldos':rho})
totspectrumoutput = pd.DataFrame({'E':np.real(e)})
evboutput = pd.DataFrame({'rv':rvb,'iv':ivb})

ldosoutput.to_csv('ldos' + name + '.txt',columns=['x','y','ldos'])
totspectrumoutput.to_csv('totspectrum' + name + '.txt',columns=['E'])
evboutput.to_csv('evb' + name + '.txt',columns=['rv','iv'])


