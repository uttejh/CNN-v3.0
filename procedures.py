import numpy
# import pyopencl as cl 
import os
from numpy import array
# os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
# os.environ['PYOPENCL_CTX'] = '1'


class Procedures:
	def __init__(self):
		self.bla = []

	# Convolution Layer filters initialization
	@staticmethod
	def initFilters(filternum,n_in,n_out,fsize,zindex):
		filters = []
		w_bound = numpy.sqrt(6./float(n_in+n_out))

		filters = numpy.random.uniform(-w_bound,w_bound,(filternum,zindex, fsize,fsize))

		return filters