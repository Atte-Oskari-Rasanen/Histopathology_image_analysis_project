import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2hed, hed2rgb
from deconvolution import Deconvolution

from PIL import Image

#from scripts import deconvolution
#from scripts.ColorDeconvolution import ColorDeconvolution



import histomicstk as htk

# Example IHC image
#ihc_rgb = data.immunohistochemistry()

ihc_rgb = Image.open("/home/atte/Pictures/testim.tif");
ihc_rgb.show();

# Declare an instance of Deconvolution, with image loaded and with color basis defining what layers are interesting
decimg = Deconvolution(image=ihc_rgb, basis=[[1, 0.1, 0.2], [0, 0.1, 0.8]])

# Constructs new PIL Images, with different color layers
layer1, layer2 = decimg.out_images(mode=[1, 2])

print(ihc_rgb.format)
print(ihc_rgb.size)
print(ihc_rgb.mode)

np_img = np.array(ihc_rgb)

from numpy import asarray

npd = asarray(ihc_rgb)

print(np_img.shape)
print(npd)  


from python_utils import *
import python_utils as utils
m = utils.convert_image_to_matrix(ihc_rgb)

ihc_rhb_dc = histomicstk.preprocessing.color_deconvolution.color_convolution(ihc_rgb, w, I_0=None)
#Try to get the stain matrix automatically (needed for the stain separation) via histomicstsk 
##########

# create stain to color map
stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
print('stain_color_map:', stain_color_map, sep='\n')

# specify stains of input image
stains = ['hematoxylin',  # nuclei stain
          'eosin',        # cytoplasm stain
          'null']         # set to null if input contains only two stains

# create stain matrix
W = np.array([stain_color_map[st] for st in stains]).T

# create initial stain matrix
W_init = W[:, :2]

# Compute stain matrix adaptively
sparsity_factor = 0.5

I_0 = 255
im_sda = htk.preprocessing.color_conversion.rgb_to_sda(imInput, I_0)
W_est = htk.preprocessing.color_deconvolution.separate_stains_xu_snmf(
    im_sda, W_init, sparsity_factor,
)

# perform sparse color deconvolution
imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(
    imInput,
    htk.preprocessing.color_deconvolution.complement_stain_matrix(W_est),
    I_0,
)

print('Estimated stain colors (in rows):', W_est.T, sep='\n')

# Display results
for i in 0, 1:
    plt.figure()
    plt.imshow(imDeconvolved.Stains[:, :, i])
    _ = plt.title(stains[i], fontsize=titlesize)

#########


# Separate the stains from the IHC image
ihc_sep = separate_stains(ihc_rgb)
ihc_hed = rgb2hed(ihc_rgb)

# Create an RGB image for each of the stains
null = np.zeros_like(ihc_hed[:, :, 0])
ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

# Display
fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(ihc_rgb)
ax[0].set_title("Original image")

ax[1].imshow(ihc_h)
ax[1].set_title("Hematoxylin")

ax[2].imshow(ihc_e)
ax[2].set_title("Eosin")  # Note that there is no Eosin stain in this image

ax[3].imshow(ihc_d)
ax[3].set_title("DAB")

for a in ax.ravel():
    a.axis('off')

fig.tight_layout()
