import numpy
# import pyopencl as cl 
import os
from numpy import array
# os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
# os.environ['PYOPENCL_CTX'] = '1'


class Procedures:
	def __init__(self):
		self.bla = []

	# Convolution Layer filters/weights initialization
	@staticmethod
	def initFilters(filternum,n_in,n_out,fsize,zindex):
		filters = []
		w_bound = numpy.sqrt(6./float(n_in+n_out))

		filters = numpy.random.uniform(-w_bound,w_bound,(filternum,zindex, fsize,fsize))

		return filters

	@staticmethod
	def convolution(x, w, bias, num, order):
		kernelsource = """
		__kernel void convolute(
		    __global double* a,
		    __global double* b,
		    __global double* c,
		    const unsigned int M,
		    const unsigned int N,
		    float bias)
		{
		    int row = get_global_id(0); 
		    int col = get_global_id(1); 

		    int receptive_col;
		    int fil_col = 0;
		    int k;
		    int receptive_row;
		    int fil_row;
		    float temp=0.0;
			
			/* Each row must end at M-N+1
			e.g - for 5*5 i/p with 3*3 filter.
			The filter must stop before M-N+1 = 3 rd so that from there (3rd) it will increment N times resulting
			in [(M-N+1)  + N ]= M + 1 (An array starts from 0 so we add 1).
			Going From TOP to BOTTOM*/

			if(row < (M-N+1))
			{		
				// Applying it from LEFT TO RIGHT		
				if(col < (M-N+1))
				{

					// Receptive Field's row. Dimensions same as filters = N*N
					receptive_row = row;

					// Filter's row. Dim = N*N
					fil_row = 0;
					temp = 0.0;

					// Looping N times so that we can move from TOP to BOTTOM 
					for(k=0;k<N;k++)
					{
						// Looping N times LEFT to RIGHT
						fil_col = 0;
						for(receptive_col=col;receptive_col<N+col;receptive_col++)
						{
							// a consists of N*N Receptive Field and b - Filter - N*N
							// adding the multiplied values with each iteration until N*N times and
							// then reinitializing temp to 0
							temp += a[receptive_row*M + receptive_col] * b[fil_row*N + fil_col];
							fil_col += 1;
					
						}
						fil_row = fil_row + 1;
						receptive_row = receptive_row+1;
					}
					// assign dot product(receptive field, filter) to C
					// SIGMA(W*X) + B
					//if(isnan(temp) || isinf(temp)){
					//	c[row*(M-N+1) + col] = 1.0;
					//}else{
					//	c[row*(M-N+1) + col] = temp + bias;
					//}
					c[row*(M-N+1) + col] = temp + bias;
				}

			}


		    
		}
		"""
		context = cl.create_some_context()
		queue = cl.CommandQueue(context)
		program = cl.Program(context, kernelsource).build()
		
		F_order = 3
		
		out_order = (order - F_order + 1)

		

		convolute = program.convolute
		convolute.set_scalar_arg_dtypes([None, None, None, numpy.uint32, numpy.uint32, numpy.float32])
		out = []

		noOffilters = len(w)

		for img in range(num):
			if (num == 1):
				h_a = x
			else:
				h_a = x[img]

			d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
			# Convoluting Image with each filter
			for filt in range(noOffilters):
				# Passing each filter
				h_b = w[filt]
				# h_b = numpy.ones((3,3)).astype(numpy.float32)
				d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)

				h_d = numpy.empty((out_order,out_order))
				d_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_d.nbytes)

				convolute(queue, (order,order), None, d_a, d_b, d_d, order, F_order, bias)
				queue.finish()
				cl.enqueue_copy(queue, h_d, d_d)

				# appending output of convolution with each filter
				out.append(h_d)		
		
		return out