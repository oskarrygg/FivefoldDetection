import numpy as np
import pyxem as pxm
import hyperspy.api as hs
import matplotlib.pyplot as plt
import cv2

from scipy import ndimage as ndimage
from skimage.util import random_noise
from skimage import feature
from skimage import measure
from skimage.transform import rotate



def create_mask_library(signal, cropped_central_region, threshold):
	sig = signal.data
	r_masks = np.zeros_like(sig, dtype = float)
	n_masks = np.zeros_like(sig, dtype = float)
	mask_lib = np.zeros_like(sig, dtype = float)

	left, right, top, bottom = cropped_central_region
	left, right, top, bottom = int(left), int(right), int(top), int(bottom)	

	for i in range(len(sig[0,])):
		for j in range(len(sig[:,0])):
			contours = measure.find_contours(sig[i,j], threshold)
			r_mask = np.zeros_like(sig[i,j], dtype = 'float')
			for contour in contours:
				r_mask[np.round(contour[:,0]).astype('int'), np.round(contour[:,1]).astype('int')] = 1
			r_mask = ndimage.binary_fill_holes(r_mask).astype(float)
			
			central_disc = measure.find_contours(r_mask[top:bottom,left:right], 0.02)
			n_mask = np.zeros_like(sig[i,j], dtype = 'float')
			for contour in central_disc:
				n_mask[np.round(contour[:,0]+top).astype('int'), np.round(contour[:,1]+left).astype('int')] = 1
			n_mask = ndimage.binary_fill_holes(n_mask).astype(float)
			
			r_masks[i,j] = r_mask
			n_masks[i,j] = n_mask
			mask = r_mask - n_mask
			mask[mask==-1] = 0

			mask_lib[i,j] = mask
	return r_masks, n_masks, mask_lib


def inspect_single_mask(full_mask, central_spot, subtracted_mask, indices):
	i, j = indices
	fig, ax = plt.subplots(1,3, figsize= (12,4))
	ax[0].imshow(full_mask[i,j])
	ax[1].imshow(central_spot[i,j])
	ax[2].imshow(subtracted_mask[i,j])

	ax[0].set_title('Full mask')
	ax[1].set_title('Central spot')
	ax[2].set_title('Central spot subtracted')



def create_rotation_library(masks, rotsym_degree):
	n = rotsym_degree
	rot_angle = 360/n

	rot_lib = np.zeros_like(masks, dtype = 'float')
	summed = np.zeros_like(masks, dtype = 'float')

	for i in range(len(masks[0,])):
		for j in range(len(masks[:,0])):
			rot_lib[i,j] = rotate(masks[i,j], rot_angle)
			summed[i,j] = masks[i,j] + rot_lib[i,j]
	return summed, rot_lib


def find_nfold_symmetries(summed_masks, rotsym_degree):
	n = rotsym_degree
	reference_lib = np.zeros_like(summed_masks, dtype = 'float')
	tuple_storage = []

	for i in range(len(summed_masks[0,])):
		for j in range(len(summed_masks[:,0])):
			if np.any(summed_masks[i,j]==2):
				reference_lib[i,j] = summed_masks[i,j]
				tuple_storage.append([i,j])
				print(f'Possible {n}-fold symmetry found in pattern [{i},{j}]. Image added to reference library for inspection.')
			else:
				if i==(len(summed_masks[0,])-1) and j==(len(summed_masks[:,0])-1) and np.all(reference_lib == 0):
					print(f'No {n}-fold symmetries found :-(')
	return reference_lib, tuple_storage



def filter_spurious_overlaps(summed_masks, tuples, threshold):
	tuple_storage = []
	for i in range(len(tuples)):
		overlaps = measure.find_contours(summed_masks[tuples[i][0], tuples[i][1]], 1.9) 
		OVERLAP = False
		for ol in overlaps:
			areas = []
			ol = np.expand_dims(ol.astype(np.float32), 1)
			ol = cv2.UMat(ol)
			areas.append(cv2.contourArea(ol))
			areas = np.asarray(areas)
			if (areas.max()>threshold):
				OVERLAP = True
		if OVERLAP:
			tuple_storage.append(tuples[i])
	
	print(f'The desired symmetry was found in {len(tuples)} diffraction patterns. Of these, {len(tuples)-len(tuple_storage)} were regarded as erroneously assigned.')
	return tuple_storage




	