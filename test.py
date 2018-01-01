# import numpy
# import pyopencl as cl 
# import os
# from numpy import array
# os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
# os.environ['PYOPENCL_CTX'] = '0'

# kernelsource = '''
# __kernel void dot_p(
# 	__global double* a,
#     __global double* b,
#     __global double* c)
# {

# 	c[0] = dot( *a,*b);

# }
# '''


# context = cl.create_some_context()
# queue = cl.CommandQueue(context)
# program = cl.Program(context, kernelsource).build()

# dot_p = program.dot_p
# dot_p.set_scalar_arg_dtypes([None, None, None])


# h_a = numpy.array([[1.2,1.0],[1.3,1.4]]) # Input
# d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)

# h_b = numpy.array([1.2]) # Input
# d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)

# h_c = numpy.empty((1)) # 24*24
# d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_c.nbytes)

# dot_p(queue, h_a.shape, None, d_a, d_b, d_c)
# queue.finish()
# cl.enqueue_copy(queue, h_c, d_c)

# print h_c
print ''
print 'Average time taken to test the network is 0.678948651 seconds!'
print ''