import ctypes
from functools import partial


def oneSideTD(img1, img2, height=28, width=28):
    d1 = ctypes.CDLL("./ts.so").oneSideTD
    d1.restype = ctypes.c_double
    c_img1 = (ctypes.c_double * len(img1))(*img1)
    c_img2 = (ctypes.c_double * len(img2))(*img2)
    return d1(c_img1, c_img2, ctypes.c_int(height), ctypes.c_int(width))


def twoSideTD(img1, img2):
    d2 = ctypes.CDLL("./ts.so").twoSideTD
    d2.restype = ctypes.c_double
    return d2((ctypes.c_double * len(img1))(*img1), (ctypes.c_double * len(img2))(*img2))
