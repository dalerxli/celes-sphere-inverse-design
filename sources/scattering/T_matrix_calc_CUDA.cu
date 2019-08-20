//Routine for calculating T-matrices of ellipsoid particles using the extended boundary condition method

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include <math.h>
#include "cuda_profiler_api.h"


//lookup for legendre function
__device__ float assocLegendreFunction(int const l, int const m, float const ct, float const st, float const *plm_coeffs)
{
	float Plm = 0.0f;
	int jj=0;
	for (int lambda=l-m; lambda>=0; lambda-=2)
	{
		Plm = Plm + pow(st,m) * pow(ct,lambda) * plm_coeffs[jj*(2*LMAX+1)*(2*LMAX+1)+m*(2*LMAX+1)+l];
		jj++;
	}
	return Plm;
}


//lookup for pi function
__device__ float assocPiFunction(int const l, int const m, float const ct, float const st, float const *plm_coeffs)
{
    float Pilm = 0.0f;
    int jj=0;
    for (int lambda=l-m; lambda>=0; lambda-=2)
    {
        Pilm = Pilm + pow(st,m-1) * pow(ct,lambda) * plm_coeffs[jj*(2*LMAX+1)*(2*LMAX+1)+m*(2*LMAX+1)+l];
    }
    return Pilm;
}

//lookup for tau function
__device__ float assocTauFunction(int const l, int const m, float const ct, float const st, float const *plm_coeffs)
{
    float Taulm = 0.0f;
    int jj=0;
    for (int lambda=l-m; lambda>=0; lambda-=2)
    {
        Taulm = Taulm + (m * pow(st,m-1) * pow(ct,lambda+1) - lambda * pow(st,m+1) * pow(ct,lambda-1)) * plm_coeffs[jj*(2*LMAX+1)*(2*LMAX+1)+m*(2*LMAX+1)+l]; 
    }
    return Taulm
}

//lookup for spherical bessel functions
__device__ float sphericalBesselLookup(int const p, float const r, float const *sphBesselTable, float const rResol)
{
    float sphBessel = 0.0f;
    float rPos = r/rResol;
    int rIdx = int(rPos);
    rPos -= rIdx;
    sphBessel = spjTable[rIdx*(2*LMAX+1)+p];

    return sphBessel;
}

__global__void transitionMatrixElement()
{
    int const elementIndex = blockDim.x * blockIdx.x + threadIdx.x;
}

void mexFunction(int nlhs, mxArray *plhs[], mxArray const *prhs[])
{
    
}