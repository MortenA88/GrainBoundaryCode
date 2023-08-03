# -*- coding: utf-8 -*-
import numpy as np
def ELDOS(E,e,v):
    # Helper function for LDOS
    # Input:
    # E: energy at which the LDOS is sought
    # e: energies
    # v: eigenvectors
    #
    # Returns:
    # erho: LDOS contribution for a particular state (e,v)
    
    av = np.abs(v)**2
    eps = 1e-7
    erho = av * np.exp(-(E-e)**2/eps)
        
    return erho

def LDOS(e,v,E):
    # Calculate the local density of states
    # Input:
    # e: energies
    # v: eigenvectors
    # E: energy at which the LDOS is sought
    #
    # Output:
    # rho: the local density of states
                
    Ne = len(e)
    rhotemp = np.zeros(len(v[:,0]))
    nbands = 4
    rho = np.zeros(int(len(v[:,0])/nbands))
    for i in range(0,Ne):
        rhotemp = rhotemp + ELDOS(E,e[i],v[:,i])

    
    for n in range(0,nbands):
        rho = rho + np.real(rhotemp[n::nbands])
            
    return rho
    
def BandSorter(e,v,px,py,ed):
    # A function which filters out edge states, retaining only
    # eigenvector with significant value on the grain boundary
    # Input:
    # e: enegies
    # v: eigenvectors
    # px,py: lattice coordinates
    # ed: location of edge dislocations
    #
    # Output:
    # eout, vout: energies and eigenvectors located on the grain boundary
    # nout: 

    apy = np.array(py)
    ne = len(e)
    nbands = 4
    eout = []
    vout = []
    
    ybs = apy[ed]
    Nm = len(ybs) // 2
    Nw = len(ybs) // 8
    
    ns = np.argsort(ybs.flatten())

    for i in range(0,ne):
        #nmax = np.argmax(np.abs(v[0::nbands,i]))
        if np.abs(e[i]) >= 0.:
            vbsu = np.abs(v[0::nbands,i])
            vbsd = np.abs(v[1::nbands,i])
        else:
            vbsu = np.abs(v[2::nbands,i])
            vbsd = np.abs(v[3::nbands,i])
            
        if np.max(vbsu) > np.max(vbsd):
            vbs = vbsu
        else:
            vbs = vbsd
            
        vbs = vbs[ed]
        vbs = vbs[ns]
        #vbs = vbs[::2]
        vmax = np.max(np.abs(vbs))
        vbs = vbs / vmax
        vmid = np.max(vbs[Nm-Nw:Nm+Nw])

        
        if vmid >= 0.5 and vmax > 1e-10:
            eout.append(e[i])
            vout.append(v[:,i])

            
    
    if len(vout)>0:
        vout = np.vstack(vout)
        vout = vout.transpose()
    return eout,np.array(vout)
    
                

def BandCalculator(e,v,xa,px,py):
    # Calculate the band structure.
    # Input:
    # e: energies
    # v: eigenvectors
    # xa: location of grain boundary nodes (assumed to have same x-position)
    # px,py: lattice coordinates
    #
    # Output:
    # kout: An array of k-values corresponding to e    
    
    Ne = len(e)
    ge = 1e-3
    c = 4
    nbands = 4
    spinbands = 2
    
    
    # Get position of GB nodes
    nab = np.argwhere(np.abs(px-xa) < ge)
    nab = nab[np.argsort(py[nab].flatten())]
    Ns = len(nab) - 2*c
    Nz = 9*Ns # Zero padding
    Ns = (Ns + Nz)
    Nf = Ns//2
    
    
    kout = np.zeros((nbands,Ne))
    
    for j in range(Ne):
        l = 0
        for n in range(spinbands):
            vs = v[n::4,j]
            
            vspos = vs[nab[c:-c]]
            vspos = np.real(vspos[::2])
        
            Favs = np.fft.fft(vspos.flatten(),Ns)
            nmax = np.argmax(np.abs(Favs[:Nf]))
            kout[l,j] = nmax/Nf
            l = l+1
            
    
    return kout

def FHS(M,Fk):
    # Numerically calculate winding number of discretized band structure
    # Input:
    # M = [k,Ek,n], where k is the momentum, Ek the energy and n the band index
    # Fk = The corresponding eigenstates for a given row in M
    #
    # Output:
    # nu: the topological invariant
    
    # Number of bands
    nbands = 4
    K = M[:,0]
    E = M[:,1]
    N = M[:,2]
    print(E[E>0])
    print(N[E>0])
    print(K[E>0])
    # Length of eigenstates
    Nf = len(Fk[:,0])
    
    # Number of eigenvalues per band
    Neb = np.zeros(nbands)
    for j in range(0,nbands):
        Neb[j] = len([i for i in M[:,2] if i == j])
    Nebm = int(np.max(Neb))
    
    # Eigenstates sorted by band index
    psi = np.zeros((Nf,Nebm,nbands))*1j
    kj  = np.zeros((Nebm,nbands))
            
    
    for j in range(0,nbands):
        psi0 = Fk[:,np.argwhere(M[:,2] == j)] 
        kj0  = np.vstack(M[np.argwhere(M[:,2] == j),0]).flatten()
        nsk = np.argsort(kj0)

        for q in range(0,len(nsk)):
        
            psi[:,q,j] = psi0[:,nsk[q]].flatten()
            kj[q,j]  = kj0[nsk[q]]
            
    
    U = np.zeros((2,2,Nebm))*1j
    dU = np.zeros(Nebm)*1j
    nu0 = 1
    for i in range(0,Nebm):
        for m in range(0,2):
            for n in range(0,2):
                if i == Nebm - 1:
                    U[m,n,i] = np.dot(np.conj(psi[:,0,m]),psi[:,i,n])
                else:
                    U[m,n,i] = np.dot(np.conj(psi[:,i+1,m]),psi[:,i,n])
                
        dU[i] = np.linalg.det(np.matrix(U[:,:,i]))
        if np.abs(dU[i]) <= 1e-20:
            dU[i] = 1
        nu0 = nu0 * dU[i] / np.sqrt(dU[i] * np.conj(dU[i]))
    
    nu = np.imag(np.log(nu0))/np.pi
    
    return nu
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

