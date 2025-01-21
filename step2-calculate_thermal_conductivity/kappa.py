#!/public/software/apps/anaconda3/5.2.0/bin/python
import numpy as np
import sys

# defult to [Nt] ns NEMD simulation, with [1fs] timestep
# total number of frame is [Nf], and devided into [Nt] part
# box devided into [NL] layers

########## Parameters to change ############
NL = 20
Nt = 5
Nf = 5000
dirct = 'x'  # the direction of heat flow (Ek exchange)
plot_fit=1                  # plot T
dt = 1.0*1.0E-09        # 1ns = 1E-9 s
exclude = 2         # exclude [] layers from heat sink and source 
############################################

def split(nframe, nbin):
    infile = open("tmp.profile", 'r')

    infile.readline()
    infile.readline()
    infile.readline()

    outfile = []
    for i in range(nbin):
        outfile.append(open("tmp."+str(i+1), 'w'))

    for n in range(nframe):
        str0 = infile.readline().split() 

        for i in range(nbin):
            str1 = infile.readline().split()
            outfile[i].write(str0[0]+str1[3].rjust(10)+'\n')

    for i in range(nbin):
        outfile[i].close()

    infile.close()

def doublefit(T,source,exclude=2):
    '''fit an array with two image regions'''
    (layer, time) = T.shape
    if source == layer//2:
        TL = T[0:source+1,:]
        TR = np.vstack(
            [ T[source:layer,:], T[0,:] ]
            )
    elif source > layer//2:
        sink=source-(layer//2)
        TL = T[sink:source+1,:]
        TR = np.vstack(
            [ T[source:layer,:], T[0:sink+1,:] ]
            )
    
    else:
        sink=layer+(source-layer//2)
        TL = np.vstack(
            [ T[sink:layer,:], T[0:source+1,:]]
            )
        TR = T[source:sink+1,:]

    (lr,hr)=(exclude, layer//2 -exclude+1)
    
    SlopL=np.zeros(time)
    InteL=np.zeros(time)
    SlopR=np.zeros(time)
    InteR=np.zeros(time)
    
    #print(TL,TR)
    for i in range(time):
        yL=TL[:,i][lr:hr]
        x=np.arange(lr,hr,1)
        SlopL[i],InteL[i]=np.polyfit(x,yL,1)
        
        yR=TR[:,i][lr:hr]
        x=np.arange(lr+layer//2,hr+layer//2,1)
        SlopR[i],InteR[i]=np.polyfit(x,yR,1)

    return (TL, TR), (SlopL, SlopR), (InteL, InteR)


def calcshape(dirct):
    # assumes a 500ps NPT (500 frames) have been performed 
    # and the thermo was stored in custum style in [log.lammps] file as
    #   Step  Temp   Press    Lx   Ly    Lz     Xy   Xz    Yz 
    infile = open('log.lammps','r')
    outfile=open('box.dat','w')
    lines=infile.readlines()

    for n, line in enumerate(lines):
        L=line.strip().split()
        if ('Lx' in L) & ('Ly' in L) & ('Lz' in L):
            outfile.write('#'+line)
            [ outfile.write(lines[i]) for i in range(n+1,n+502) ]
            print('\nBox data has written to box.dat')
            break
        elif n==len(lines):
             print('\nNo data found!\n')
    outfile.close()
    infile.close()
     
    box=np.loadtxt('box.dat')

    print('Using box data in the last 400ps...')
    box_ave=np.average(box[100:,3:], axis=0)


    print("\n   {:8s}{:8s}{:8s}{:8s}{:8s}{:8s}".format('Lx','Ly','Lz','Xy','Xz','Yz'))
    [Lx,Ly,Lz,Xy,Xz,Yz] = box_ave.tolist()
    print("{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n".format(Lx,Ly,Lz,Xy,Xz,Yz))

    # formulas can be found at https://gensoft.pasteur.fr/docs/lammps/2020.03.03/Howto_triclinic.html
    a = Lx
    b = np.sqrt(
            Ly**2 + Xy**2
        )
    c = np.sqrt(
            Lz**2 + Xz**2 + Yz**2
        )
    alpha = np.arccos(
        (Xy*Xz+Ly*Yz)/(b*c)
        )
    beta = np.arccos(Xz/c)
    gamma = np.arccos(Xy/b)
    abc = np.array([
            [Lx, Xy, Xz],
            [0,Ly,Yz],
            [0,0,Lz]
        ])
    abc_inv = 2*np.pi*np.transpose(np.linalg.inv(abc))

    # COS of the angle between a*/b*/c* and x/y/z, respectively
    one = np.eye(3)
    cos_a = np.dot(one[:,0], abc_inv[:,0])/(np.linalg.norm(one[:,0])*np.linalg.norm(abc_inv[:,0]))
    cos_b = np.dot(one[:,1], abc_inv[:,1])/(np.linalg.norm(one[:,1])*np.linalg.norm(abc_inv[:,1]))
    cos_c = np.dot(one[:,2], abc_inv[:,2])/(np.linalg.norm(one[:,2])*np.linalg.norm(abc_inv[:,2]))

    dircts=('x','y','z')
    length={'x':Lx,'y':Ly,'z':Lz}
    S={'x':b*c*np.sin(alpha),'y':a*c*np.sin(beta),'z':a*b*np.sin(gamma)}
    A={'x':Ly*Lz,'y':Lx*Lz,'z':Ly*Lx}
    COS = {'x':cos_a,'y':cos_b,'z':cos_c}

    print("   {:8s}{:8s}{:8s}{:8s}{:8s}{:8s}".format('a','b','c','alpha','beta','gamma'))
    print("{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}\n".format(
        a,b,c,alpha/np.pi*180,beta/np.pi*180,gamma/np.pi*180
        ))

    if dirct in dircts:
        print("Total length:  {:.3f} A".format(length[dirct]*COS[dirct]))
        print("Plane area:    {:.3f} A^2".format(S[dirct]))
        print("Cross area:    {:.3f} A^2".format(A[dirct]))
        print("COS:           {:.3f}".format(COS[dirct]))
        
        return length[dirct], S[dirct], COS[dirct]
    else:
        print('Only x y z are supported!\n')   


def main():

    print("\nWARNING: Do not forget to change the parameters!")
    
    print("Extracting tempertures from tmp.profile...")
    split(Nf, NL)
    
    Tarray=np.zeros((NL,Nt))
    for l in range(NL):
        data=np.loadtxt('tmp.'+str(l+1))[:,1]
        #data=data.reshape((Nf//Nt,Nt))
        #Tarray[l,:]=np.average(data, axis=0)
        # xcl: 2024.01.09
        data=data.reshape((Nt, Nf//Nt))
        Tarray[l,:]=np.average(data, axis=1)
    
    print('The last 4 ns data will be used...')
    Tarray=Tarray[:,1:Nt]
    
    Tave=np.average(Tarray, axis=1)
    source=np.argmax(Tave)
    
    if source != NL//2:
        print(f"WARNING: The simulated position of heat/sink source may be wrong! ({source}/{np.argmin(Tave)})")
    
    if Nt == 5:
        with open('tmp.ave', 'w') as tem:
            tem.write("#Layer    1ns    2ns    3ns    4ns    Ave\n")
            [ 
             tem.write("{}  {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(
                i,Tarray[i,0],Tarray[i,1],Tarray[i,2],Tarray[i,3],Tave[i])) for i in range(NL) 
            ]
    
    _, Slop, _= doublefit(Tarray, source, exclude=exclude)
    Slop = (Slop[0]-Slop[1])/2
    
    T,k,b  = doublefit(np.array([Tave]).T, source, exclude=exclude) # average for all last 4ns
    
    
    if plot_fit:
        T = np.hstack([T[0][:,0], T[1][1:,0]])
        print("Ploting figure to fitT.png...")
        import matplotlib
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        (lr,hr)=(exclude, NL//2 -exclude+1)
        layer = np.arange(NL+1)
        layerL=np.arange(lr,hr)  # reset the sink ID to 0 
        layerR=np.arange(lr+NL//2,hr+NL//2)
        
        plt.figure(figsize=(4,3.2), facecolor='w')
        plt.scatter(layer,T,s=20,c='blue',marker='s',label='NEMD')
        plt.plot(layerL,k[0]*layerL+b[0],'--',c='black',label='fit')
        plt.plot(layerR,k[1]*layerR+b[1],'--',c='black')
        plt.xlim(-0.5,NL+0.5)
        plt.xlabel('Layer number')
        plt.ylabel('T (K)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('fitT.png', dpi=300)
    
    print("Loading heat flow from log.lammps...")
    log=open('log.lammps','r')
    lines=log.readlines()
    E = []
    key=['1000000','2000000','3000000','4000000','5000000']
    for line in lines:
        line=line.strip().split()
        if len(line)>0 and line[0] in key:
            E.append(float(line[4]))
    log.close()
    E=np.array(E)
    
    dE=np.array([E[i+1]-E[i] for i in range(Nt-1)])  *4184/(6.022E23)  # J, (4,)
    
    print("Calculating thermal conductivity...")
    
    # The direction of κ may not along x,y,z
    # it should along the norm vector of the plane
    # which means the direction of dT should also along
    # the norm vector of the plane. 
    # So, we have to multiply a COS factor on dT
    # This is equivariant to use dT*A, where A is the cross area
    
    dL, S, COS = calcshape(dirct)
    
    S = S*1.0E-20          # A^2 -> m^2
    dL = dL/20*1.0E-10     # A -> m,   length per layer
    
    J=dE/dt/(2*S)    # W/(m^2)
    dT=Slop/dL*COS       # K/m
    kappa=J/dT       # W/(m*K)
    
    k = (k[0][0] - k[1][0])/2
    kappa_ave=np.average(J)/(k/dL*COS)

    std = np.sqrt((np.average(kappa**2)-kappa.mean()**2)/(Nt-2))
    
    print('\nThermal conductivity κ:  {:.3f} W/(m K)'.format(kappa_ave))
    print('Standard error of κ:    {:.4f} W/(m K)\n'.format(std))
    with open('kappa.out', 'w') as out:
        out.write("COS: {:.5e} m^2\n".format(COS))
        out.write("Length per layer: {:.5e} m\n".format(dL))
        out.write("\nStep     f_1 (kJ/mol)\n")
        [ out.write("{}  {:.3f}\n".format(key[i],E[i])) for i in range(Nt) ]
        out.write('\nThermal conductivity κ:  {:.3f} W/(m K)\n'.format(kappa_ave))
        out.write('Standard error of κ:    {:.4f} W/(m K)\n'.format(std))

if __name__=="__main__":
    main()

