import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

dir = '/Users/boulanger/Downloads/Clement_CO/'
dir_sim = dir+'Simulations/'
file_in = dir + 'map_CO12_100-143_from_100x143_DATA_R1_gold+hlat_new_full_nsideI2048_nsideQU256_nsideCO1024.dat'
map_CO = np.fromfile(file-in)
hp.mollview(map_CO,max=15.,min=-1.)
plt.show()

# extract flat-sky image
# Galactic coordinates of image center
glon = 300.
glat =-20.
# pixel size arcminutes
pix_arcmin = 2.35
# image size
Nx = 768
Ny = 768
# return numpy array of the image in K km/s
CO = hp.gnomview(map_CO,coord='G',rot=[glon,glat],reso=pix_arcmin,xsize=Nx,ysize=Ny,return_projected_map=True)
plt.show()

fac_12CO = 1.48e-5 # scaling from K km/s to K CMB

# CO input model used in Simulations

# read input 12CO(1-0) healpix image Nside=128
file_path = dir+ 'Simulations/12COforSIMU.float32.bin'

# Open the binary file in binary read mode
with open(file_path, 'rb') as f:
    # Read the binary data from the file
    binary_data = f.read()

buf = np.frombuffer(binary_data, dtype=np.float32)

# procedure to upgrade buf sky map to Nside = 256
def up_grade(im,onside,FWHM=20./60./180.*np.pi):
        th,ph=hp.pix2ang(onside,np.arange(12*onside*onside))
        val=hp.get_interp_val(im,th,ph)
        val=hp.smoothing(val,FWHM)
        return(val)


# Simulations 
    

f0 = dir_sim + 'map_CO12_100-143_from_100x143_000_R1_gold+hlat_new_nsideI2048_nsideQU256_nsideCO1024.dat'
f1 = dir_sim + 'map_CO12_100-143_from_100x143_000_R1_gold+hlat_new_hm1_nsideI2048_nsideQU256_nsideCO1024.dat'
f2 = dir_sim + 'map_CO12_100-143_from_100x143_000_R1_gold+hlat_new_hm2_nsideI2048_nsideQU256_nsideCO1024.dat'
map0 = np.fromfile(f0)
map1 = np.fromfile(f1)
map2 = np.fromfile(f2)

map0l = hp.ud_grade(map0,256,order_out='RING')
dif0 = map0l-CO_mod

dif = 0.5*(map2-map1)
res= hp.gnomview(dif,coord='G',rot=[glon,glat],reso=pix_arcmin,xsize=Nx,ysize=Ny,return_projected_map=True)
np.std(res[0:201,468:768])

plt.imshow(res,vmin=-2e-4,vmax=2e-4)
plt.colorbar()
plt.show()

plt.imshow(CO*fac_12CO,vmin=-2e-4,vmax=2e-4)
plt.colorbar()
plt.show()
np.std(CO[0:201,468:768])*fac_12CO
