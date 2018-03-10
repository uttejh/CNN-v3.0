# CNN-v3.0

A Convolutional Neural Network used for the recognition of Handwritten characters (numbers) using Deep Learning. The system was
purely implemented from the scratch in python without using any existing tools such as Tensorflow, Matlab, Keras, etc... and the performance was parallelized with the help of OpenCl. The architecture is
based on the <a href="http://neuralnetworksanddeeplearning.com/">book</a> by <a href="http://michaelnielsen.org/">Michael Nielsen</a>.
      The system was trained using very small datasets for around 30000 epochs and has achieved an accuracy of 79%.

# Built-With
<ul>
  <li>Pyhton</li>
  <li><a href="https://www.khronos.org/opencl/">Opencl</a> - open standard for parallel programming of heterogeneous systems</li>
</ul>

# Changes
The changes from the previous version (v2) are as follows:-
<ul>
  <li>The activation function is changed from ReLu to Sigmoid</li>
  <li>Normalisation layers were added</li>
  <li>Redundant functions were removed</li>
</ul>

# Authors
<ul>
  <li><a href="https://github.com/uttejh">Uttejh reddy</a></li>
  <li><a href="https://github.com/sumeesha">Sumeesha marakani</a></li>
  <li><a href="https://github.com/nprithviraj24">Prithvi raj</a></li>
</ul>

# License
<li>The project uses open-source software</li>

# Acknowledgements
<ul>
  <li><a href="http://neuralnetworksanddeeplearning.com/chap6.html">Neural Networks</a> - The base for our architecture.</li>
  <li><a href="http://cs231n.github.io/optimization-1/">CS231</a> - Visually understanding the gradient</li>
  <li><a href="https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/">Backpropagation</a> - Practical example</li>
  <li><a href="http://jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/">DeepGrid</a> - understanding how NN works</li>
</ul>
