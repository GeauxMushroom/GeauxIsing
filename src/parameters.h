#include "COPYING"


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


// SM per GPU
// should later implement GD = func (prop.multiProcessorCount);

// GD - blocksPerGrid, must be even
// BD - threadsPerBlock
// when modifing "GD", should also update "GD_HF",

#define GD 32
#define GD_HF 16

#define TperB 144
// checkerboard 3D block
#define BDx0 L_HF
#define BDy0 L
#define BDz0 2
// BDZ0 == 1 fail ???
//
// 1D block
#define BDx3 TperB
#define BDy3 1
#define BDz3 1


#define MPIRANK_PER_NODE 2

#endif /* PARAMETERS_H */

