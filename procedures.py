import numpy
import pyopencl as cl 
import os
from numpy import array
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1' # uses device[0] to run everthing.(0-GPU,1-CPU;IN MY CASE)


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


	# weights initialization of HL and FC
	@staticmethod
	def initWeights(inp,out):
		weights = []
		w_bound = numpy.sqrt(6./float(inp+out))

		weights = numpy.random.uniform(-w_bound,w_bound,(inp,out))

		return weights


	# Bias initialization
	@staticmethod
	def initBias(num):
		bias = []
		bias = numpy.random.uniform(0.,1.,(num))

		return bias


	@staticmethod
	def convolution(x, w, numFilters, zindex, fsize, bias): # x, w, bias, num, order
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

		convolute = program.convolute
		convolute.set_scalar_arg_dtypes([None, None, None, numpy.uint32, numpy.uint32, numpy.float32])

		out = []

		order = x.shape[2]
		out_order = order - fsize + 1


		for i in range(numFilters):
			temp_out = []
			for j in range(zindex):

				h_a = x[j] # Input
				d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)

				h_b = w[i][j] # Filter / Weight 
				d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)

				h_c = numpy.empty((out_order,out_order)) # 24*24
				d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_c.nbytes)

				convolute(queue, (order,order), None, d_a, d_b, d_c, order, fsize, bias)
				queue.finish()
				cl.enqueue_copy(queue, h_c, d_c)

				# appending output of convolution with each filter
				temp_out.append(h_c)

			# Converting 20*8*8 into 8*8 (example) (3D->2D)
			temp_var = numpy.sum(temp_out,axis=0)
			
			out.append(temp_var)

		return out

	@staticmethod
	def pooling(x, num):
		kernelsource = """
			__kernel void pool(
		    __global double* A,
		    __global double* B,
		    __global double* C,
		    const unsigned int N)
		    {
				int i = get_global_id(0);
			    int j = get_global_id(1);

			    int index1;
			    int index2;
				
				double t1,t2,t3,t4,t5,t6;
			    if ((i < N-1) && (i%2 == 0))
			    {
					if ((j < N-1) && (j%2 == 0))
				    {
						t1 = A[i*N + j];
						t2 = A[i*N + j+1];
						t3 = A[(i+1)*N + j];
						t4 = A[(i+1)*N + j+1];
						if(t1>t2)
						{
							t5 = t1;
							index1 = i*N + j;
						}
						else{
							t5 = t2;
							index1 = i*N + j + 1;
						}

						if(t3>t4)
						{
							t6 = t3;
							index2 = (i+1)*N + j;
						}
						else{
							t6 = t4;
							index2 = (i+1)*N + j+1;
						}
						int x = (i/2);
						int y = (j/2);
						if(t5>t6)
						{
							B[x*(N/2) + y] = t5;
							C[x*(N/2) + y] = index1;
						}else{
							B[x*(N/2) + y] = t6;
							C[x*(N/2) + y] = index2;
						}
				    }
			    }
		    }
		"""

		context = cl.create_some_context()
		queue = cl.CommandQueue(context)
		program = cl.Program(context, kernelsource).build()

		order = x.shape[2]
		out_order = (order/2)

		pool = program.pool
		pool.set_scalar_arg_dtypes([None,None,None,numpy.uint32])

		pool_out = []
		index = []

		for it in range(num):
			h_a = x[it]
			d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)

			h_b = numpy.empty((out_order,out_order))
			d_b = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_b.nbytes)

			h_c = numpy.empty((out_order,out_order))
			d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_c.nbytes)

			pool(queue, (order, order), None, d_a, d_b, d_c, order)
			queue.finish()
			cl.enqueue_copy(queue, h_b, d_b)
			cl.enqueue_copy(queue, h_c, d_c)

			pool_out.append(h_b)
			index.append(h_c)

		return pool_out,index
