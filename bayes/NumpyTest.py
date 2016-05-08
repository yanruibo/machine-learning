'''
Created on Nov 5, 2015

@author: yanruibo
'''
import numpy as np
from scipy.io.matlab.miobase import arr_dtype_number
if __name__ == '__main__':
    
    
    arr = np.reshape(range(15), (3, 5))
    print arr
    print arr[:,-1]
    ans = np.delete(arr, [len(arr[0])-1], axis=1)
    print ans
    