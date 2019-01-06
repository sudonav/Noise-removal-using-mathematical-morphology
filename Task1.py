
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


noisy_image = cv.imread("noise.jpg", 0)


# In[3]:


def apply_erosion(input_image, structuring_element_shape=(3,3)):
    top = bottom = np.floor(structuring_element_shape[0]/2).astype(int)
    left = right = np.floor(structuring_element_shape[1]/2).astype(int)
    padded_image = np.pad(input_image, ((top,bottom),(left,right)), 'constant').astype(int)
    structuring_element = np.ones(structuring_element_shape, np.uint8)
    output_image = np.ones(input_image.shape)
    for i in range(len(input_image)):
        for j in range(len(input_image[0])):
            patch = padded_image[i:i+structuring_element_shape[0],j:j+structuring_element_shape[1]]
            if(np.array_equal(patch, structuring_element)):
                output_image[i][j] = 1
            else:
                output_image[i][j] = 0
    return output_image


# In[4]:


def hints(img1, img2):
    return (np.ravel(img1) == np.ravel(img2)).any()


# In[5]:


def apply_dilation(input_image, structuring_element_shape=(3,3)):
    top = bottom = np.floor(structuring_element_shape[0]/2).astype(int)
    left = right = np.floor(structuring_element_shape[1]/2).astype(int)
    padded_image = np.pad(input_image, ((top,bottom),(left,right)), 'constant').astype(int)
    structuring_element = np.ones(structuring_element_shape, np.uint8)
    output_image = np.ones(input_image.shape)
    for i in range(len(input_image)):
        for j in range(len(input_image[0])):
            patch = padded_image[i:i+structuring_element_shape[0],j:j+structuring_element_shape[1]]
            if(hints(patch, structuring_element)):
                output_image[i][j] = 1
            else:
                output_image[i][j] = 0
    return output_image


# In[6]:


def apply_opening(input_image, structuring_element_shape=(3,3)):
    eroded_image = apply_erosion(input_image, structuring_element_shape)
    return apply_dilation(eroded_image, structuring_element_shape)


# In[7]:


def apply_closing(input_image, structuring_element_shape=(3,3)):
    dilated_image = apply_dilation(input_image, structuring_element_shape)
    return apply_erosion(dilated_image, structuring_element_shape)


# In[8]:


def open_close_algorithm(input_image, structuring_element_shape=(3,3)):
    opened_image = apply_opening(input_image, structuring_element_shape)
    return apply_closing(opened_image, structuring_element_shape)


# In[9]:


def close_open_algorithm(input_image, structuring_element_shape=(3,3)):
    closed_image = apply_closing(input_image, structuring_element_shape)
    return apply_opening(closed_image, structuring_element_shape)


# In[10]:


output_algorithm1 = open_close_algorithm((noisy_image/255).astype(int), (3,3))
cv.imwrite("res_noise1.jpg",output_algorithm1*255)


# In[11]:


output_algorithm2 = close_open_algorithm((noisy_image/255).astype(int), (3,3))
cv.imwrite("res_noise2.jpg",output_algorithm2*255)


# In[12]:


boundary_algorithm1 = output_algorithm1 - (apply_erosion(output_algorithm1, (3,3)))
cv.imwrite("res_bound1.jpg",boundary_algorithm1*255)


# In[13]:


boundary_algorithm2 = output_algorithm2 - (apply_erosion(output_algorithm2, (3,3)))
cv.imwrite("res_bound2.jpg",boundary_algorithm2*255)

