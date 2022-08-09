


# Diffusion equation (Implicit Crank-Nicolson)

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin
L=1.0
T=1.0
alpha=0.001
Nx=200
Nt=20
t,dt=np.linspace(0.0,T,Nt+1,retstep=True)# Mesh points in time
x,dx=np.linspace(0.0,L,Nx+1,retstep=True) # Mesh points in space
lam=(alpha)*(dt)/(dx**2)
print(lam)

s='t%f'
u=np.zeros(Nx+1) # unknown u at new time level
u_n=np.zeros(Nx+1) # u at the previous time level
# Data structure for the linear system
A=np.zeros((Nx+1,Nx+1))
B=np.zeros((Nx+1,Nx+1))
b=np.zeros(Nx+1)
# Tridiagonal matrix of Backward euler method scheme
for i in range(1,Nx):
    A[i,i-1]=-lam/2.
    A[i,i+1]=-lam/2.
    A[i,i]=1.+lam
# First and last row of matrix A
    A[0,0]=A[Nx,Nx]=(1.+lam)
    A[0,1]=A[Nx,Nx-1]=-lam/2.
print (A)
# Tridiagonal matrix of crank Nicolson Scheme(Matrix B)
for i in range(1,Nx):
    B[i,i-1]=lam/2.
    B[i,i+1]=lam/2.
    B[i,i]=1.-lam
# First and last row of matrix B
    B[0,0]=B[Nx,Nx]=(1.-lam)
    B[0,1]=B[Nx,Nx-1]=lam/2.
print ("\n",B)  

# Initial distribution,given by the following equation
sig=0.1
def I(x):
    return(1./np.sqrt(2*np.pi*sig))*np.exp(-(x-L/2)**2/(2*sig**2))
           
# Set initial condition u(x,0)=I(x)
for i in range(0,Nx+1):
    u_n[i]=I(x[i])
for t in range(0,Nt):
    # Compute b and solve linear system
    for i in range(1,Nx):
        b[i]=u_n[i]
    b[0]=u_n[0]+lam*0. #(lam)*times the boundary value at beginning
    b[Nx]=u_n[0]+lam*0. # (lam)* boundary value at the end
    u[:]=lin.solve(A,B.dot(b))
    plt.plot(x,u,label=s%t)
    plt.legend()
    # Update u_n before next step
    u_n[:]=u
plt.show()























           
                
         
            
