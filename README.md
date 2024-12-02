In this assignment I provided several implementations of matrix multiplication algorithms, both serial and parallel versions and analyze their performance.
First, I explored several options for different implementations of square matrix multiplications,
resulting from the possible permutations in the order in which multiplication loops i, j, k are performed. 
I implemented matrix multiplication variants corresponding to all 6 permutations of (i,j,k): i-j-k, i-k-j, j-i-k, j-k-i, k-i-j, k-j-i. 

I also implemented the blocked matrix multiplication algorithm in serial and parallel version using OpenMP, 
taking into account also the case when size of the matrix is not evenly divisible to the block size.

In the 2 word documents,I determined which version has the best serial time and which version the best parallel time and which is the best block size for my computer,
making several measurements.
