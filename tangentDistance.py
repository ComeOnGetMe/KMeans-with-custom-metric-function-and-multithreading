import ctypes


def oneSideTD(img1, img2):
    d1 = ctypes.CDLL("./ts.so").oneSideTD
    d1.restype = ctypes.c_double
    return d1((ctypes.c_double * len(img1))(*img1), (ctypes.c_double * len(img2))(*img2))


def twoSideTD(img1, img2):
    d2 = ctypes.CDLL("./ts.so").twoSideTD
    d2.restype = ctypes.c_double
    return d2((ctypes.c_double * len(img1))(*img1), (ctypes.c_double * len(img2))(*img2))
