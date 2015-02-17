#ifndef PARAMETERS_H
#define PARAMETERS_H



#define BETA_LOW 0.1f
#define BETA_HIGH 1.8f

// external field
#define H 0.1f
//#define H 1.0f

// randomly initialize J
//#define RANDJ
#define RANDS


// using the same random number for every spins integrated in a world
//#define SHARERAND







// iteration parameters
// wall time estimation for 16 realizations, 16^3 cubic lattice, 2,000,000 MCS
// NBETA * (16 ^ 3) * 16 * (2 * 10^6) * (50PS/spin) = 170 seconds


/*
#define ITER_WARMUP          4000
#define ITER_WARMUP_KERN     1000
#define ITER_WARMUP_KERNFUNC 200
#define ITER_SWAP            4000
#define ITER_SWAP_KERN       1000
#define ITER_SWAP_KERNFUNC   10
*/

#define ITER_WARMUP          0
#define ITER_WARMUP_KERN     0
#define ITER_WARMUP_KERNFUNC 0
#define ITER_SWAP            1000000
#define ITER_SWAP_KERN       1000
#define ITER_SWAP_KERNFUNC   10



#define REC_SIZE ( ITER_SWAP / ITER_SWAP_KERN )





// lattice size, must be even

#define L 12
#define L_HF (L / 2)
#define SZ_CUBE (L * L * L)
#define SZ_CUBE_HF (SZ_CUBE / 2)

#define CUBEIDX(z, y, x) ((L * L * (z)) + (L * (y)) + (x)) 




// blocksPerGrid, must be even
#define GD 28
// half of GD
#define GD_HF 14
// threadsPerBlock
#define BD 144





#define MPIRANK_PER_NODE 2

#endif /* PARAMETERS_H */

