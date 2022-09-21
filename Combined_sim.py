
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.special import binom
import math

## Parameters for creation of meshgrid
N=2
offsetAngle = 0
amplitude = 1
Rmag = 1
scl = 6.5
## Creates rectangular mesh;
numPoints = 50
x_mesh,y_mesh = np.meshgrid(np.linspace(-1,1,numPoints), np.linspace(-1,1,numPoints))
r = np.sqrt(np.power(x_mesh,2) + np.power(y_mesh,2))
x_mesh = x_mesh[r<=Rmag]
y_mesh = y_mesh[r<=Rmag]


#---------------------------#
#       Functions           #
#---------------------------#
#Creates a vector given a point (x, y), in the case of meshgrid creates whole field
def nPoleComponent(n,x,y,amplitude, angle,Rref = 1):
    #Complex plane
    z = x+y*1j
    #Complex coefficients
    Cn = amplitude*(1+0*1j)
    #Creating By + iBy
    ByBx = Cn*np.power(z/Rref,n-1)+0.0000001*np.cos(6*np.pi*z)+0.0000001*np.sin(6*np.pi*z)
    #Rotation
    ByBx = ByBx*np.exp(n*angle*1j)
    #Returning By and Bx
    return(ByBx.real, ByBx.imag)

#Returns all the points of the rand of a circle
def pointsOfCircle(x1,y1,r,stepsize):
    out_x = r*np.sin(np.linspace(0,2*np.pi,stepsize)) + x1
    out_y =r*np.cos(np.linspace(0,2*np.pi,stepsize)) + y1
    return(out_x, out_y)


#Returns all the points inside the circle
def pointsInCircle(x_point, y_point, r, r_step): #Not working
    x = []
    y = []
    for x in range(r_step):
        x  = np.append(x, pointsOfCircle(x_point,y_point,r,60)[0])
        y  = np.append(y, pointsOfCircle(x_point,y_point,r,60)[1])
    return x, y

#Returns the B_r array
def B_r(x,y,Bx,By):
    phi = np.arctan2(y,x)
    B_r = np.empty_like(Bx)
    for n in range(len(Bx)):
        B_r[n] = Bx[n]*np.cos(phi[n]) + By[n]*np.sin(phi[n])
    return B_r

#Function used to create several circles to "measure" from
def CircleOfCircles(x_point,y_point,r_large,r_small,stepsize,n_circles):
    if n_circles <= 1:
        (out_x,out_y) = pointsOfCircle(x_point,y_point,r_large,stepsize)
        out_pos = 1j*x_point+y_point
    else:
        (x,y) = pointsOfCircle(x_point,y_point,r_large,n_circles + 1)
        out_x = np.zeros(0)
        out_y = np.zeros(0)
        out_pos = np.zeros(0)
        for n in range(len(x)-1):
            out_x = np.append(out_x,pointsOfCircle(x[n], y[n], r_small, stepsize)[0])
            out_y = np.append(out_y,pointsOfCircle(x[n], y[n], r_small, stepsize)[1])
            out_pos = np.append(out_pos, 1j*x[n]+y[n])
    return out_x, out_y, out_pos

def Fourier_norm(t): #Function to return array normalized to the main field component N, and removes the first from and including N
    sp = np.fft.fft((t))
    freq = np.fft.fftfreq(t.shape[-1])
    sp = sp[N:15+N] #Only take multipoles from main component and following
    
    sp = 1e4/sp[0]*sp #Scale the multipoles
    """
    sp_ind = np.argmax(sp)
    sp_norm = [(10e4*x)/sp[sp_ind] for x in sp ]
    """
    return sp, freq

# Under follows the functions for combining several measurements
# as explained in "Combining rotating coil measurements" 

#Make matrix Wi
def matrixWi(N,K,zi,r0,rc):
    result = np.zeros((N,K))*1j
    for n in range(1,N+1):
        for k in range(n,K+1):  
            result[n-1,k-1] = binom(k-1,k-n)*pow((zi/r0),(k-n))*pow((rc/r0),(n-1))   
    return result

#this function computes one whole matrix M
def MatrixM(N,K,r0,rc,z_pos):
    #number of positions
    if not np.shape(z_pos):
        I = 1
    else:
        I = np.shape(z_pos)[0]
    #allocate space for matrix M
    M = np.zeros((I*N,K))*1j
    try:
        for i,zi in enumerate(z_pos): 

            matrix = matrixWi(N,K,zi,r0,rc)
            M[i*N:(i+1)*N,:] = matrix
    except TypeError:
        print(z_pos)
        matrix = matrixWi(N,K,z_pos,r0,rc)
    return M

def MatrixC(t):
    out = np.zeros(0)
    for x in range(0, num_circles):
        sp = np.fft.fft((t[num_samples*x:num_samples*x+num_samples]))
        #sp = sp[0:15]
        out = np.append(out,sp)
    return out


#--------------------------#
# PLOTTING MAGNETIC FIELD  #
#--------------------------#
 
#Circle paramters
x_point = 0.0
y_point = 0.0
circle_r = 0.3
large_circle_r = 0.65
num_circles = 10
num_samples = 60 #We want the 15 first multipoles and due to Nyquist-Shannon theorem 30 samples should be enough


#Calling function to get the field
(By_whole, Bx_whole) = nPoleComponent(N,x_mesh,y_mesh,amplitude,offsetAngle,1)

#Next lines create circle coordinates and calculate the vectors along the rand of circles
(X,Y, pos) = CircleOfCircles(x_point,y_point,large_circle_r,circle_r,num_samples,num_circles)
(By1, Bx1) = nPoleComponent(N,X,Y,amplitude,offsetAngle,1)
 
(X_ref,Y_ref,pos_ref) = CircleOfCircles(x_point,y_point,large_circle_r,circle_r,num_samples,1)
(By1_ref, Bx1_ref) = nPoleComponent(N,X_ref,Y_ref,amplitude,offsetAngle,1)

 #Plotting the magnetic field and our circles along with their origin
fig, ax = plt.subplots()
ax.use_sticky_edges = False
ax.margins(0.07)
ax.axis('on')
ax.tick_params(direction = 'in',bottom=True,top=True,left=True,right=True)
p = ax.quiver(x_mesh,y_mesh,Bx_whole,By_whole, units = 'xy', cmap = plt.cm.winter,zorder=2,
          width=0.007, headwidth=3.,scale=scl ,headlength=4.5,color='grey')
q = ax.quiver(X,Y,Bx1,By1, units = 'xy', cmap = plt.cm.winter,zorder=2,
          width=0.007, headwidth=1., scale=scl ,headlength=4.5,color='red')
plt.plot(np.imag(pos),np.real(pos),'o')
plt.plot(X,Y, '.', color='black')
plt.show()

#Plotting the Fourier transform


t_ref = (B_r(X_ref, Y_ref , Bx1_ref, By1_ref))
sp = np.fft.fft(t_ref)
sp = sp[N:15+N] #Only take multipoles from main component and following
sp = 1e4/sp[0]*sp #Scale the multipoles


plt.subplot(131)
plt.plot(np.linspace(0,2*np.pi,num_samples),t_ref)
plt.title('Curve from ref vec')
plt.subplot(132)

plt.stem(np.linspace(0,14,15), np.abs(np.real(sp[0:15])), linefmt='C0-',
         markerfmt=" ", basefmt="-b")
plt.legend(['Reference: Real'])
plt.subplot(133)
plt.stem(np.linspace(0,14,15), np.abs(np.imag(sp[0:15])), linefmt='C1-' ,
         markerfmt=" ", basefmt="-")
plt.legend(['Reference: Imag'])

plt.show()

t = (B_r(X,Y,Bx1, By1))
C = MatrixC(t)

M = MatrixM(num_samples,num_samples,large_circle_r,circle_r,pos)
Cp,res,rank,s = np.linalg.lstsq(M,C,rcond=None)


Cp_norm = Cp[N:15+N] #Only take multipoles from main component and following
Cp_norm = 1e4/Cp_norm[0]*Cp_norm #Scale the multipoles

plt.subplot(131)
plt.stem(np.linspace(0,14,15), np.abs((sp[:15])),  linefmt='C0-',
         markerfmt=" ", basefmt="-b")
plt.legend(['Ref'])
plt.subplot(132)

plt.stem(np.linspace(0,14,15), np.abs((Cp_norm[:15])), linefmt='C1-',
         markerfmt=" ", basefmt="-")
plt.legend(['Reconstructed'])
plt.subplot(133)

print(Cp_norm[1]/np.real(Cp_norm[1]))
print(sp[1]/np.real(sp[1]))
plt.plot(np.linspace(0,2*np.pi,int(num_samples)),np.fft.ifft(Cp))
plt.legend(['Reconstructed curve'])
plt.show()


