#include "cufft.h"

extern "C"
void getdevice(int *GPU_N);

extern "C"
struct Source
{
	int s_iz, s_ix, s_iy, r_iz, *r_ix, *r_iy, r_n;
};

extern "C"
struct MultiGPU
{
	// cufft handle for forward and backward propagation
	cufftHandle PLAN_FORWARD;
	cufftHandle PLAN_BACKWARD;

	// host pagelock memory (variables needs to cudaMemcpyDeviceToHost)
	float *pxx, *pyy, *pzz, *pxy, *pxz, *pyz, *vx, *vy, *vz;

	float *record,*record2,*record3;
	

	// device global memory (variables needs to cudaMemcpyHostToDevice)
	int *d_r_ix, *d_r_iy;
	float *d_rik;

	float *d_velp, *d_gama_p,*d_vels, *d_gama_s, *d_rho;

	float *d_pxx, *d_pyy, *d_pzz, *d_pxy, *d_pxz, *d_pyz, *d_vx, *d_vy, *d_vz;

	float *d_gammax, *d_alphax, *d_Omegax, *d_a_x, *d_b_x;
	float *d_gammay, *d_alphay, *d_Omegay, *d_a_y, *d_b_y;
	float *d_gammaz, *d_alphaz, *d_Omegaz, *d_a_z, *d_b_z;
	float *d_phi_vx_xx, *d_phi_vy_yx, *d_phi_vz_zx;
	float *d_phi_vx_xy, *d_phi_vy_yy, *d_phi_vz_zy;
	float *d_phi_vx_xz, *d_phi_vy_yz, *d_phi_vz_zz;
	float *d_phi_vx_z, *d_phi_vz_x, *d_phi_vx_y, *d_phi_vy_x, *d_phi_vy_z, *d_phi_vz_y;

	float *d_phi_pxx_x, *d_phi_pxy_y, *d_phi_pxz_z;
	float *d_phi_pxy_x, *d_phi_pyy_y, *d_phi_pyz_z;
	float *d_phi_pxz_x, *d_phi_pyz_y, *d_phi_pzz_z;
	

	cufftComplex *d_inx, *d_iny, *d_inz, *d_in_pxx, *d_in_pyy, *d_in_pzz, *d_in_pxy, *d_in_pxz, *d_in_pyz;
	cufftComplex *d_outx, *d_outy, *d_outz, *d_outpxx, *d_outpyy, *d_outpzz, *d_outpxy, *d_outpxz, *d_outpyz;


	float *d_kx, *d_ky, *d_kz, *d_k;

	///////////
	cufftComplex *d_dvx, *d_dvy, *d_dvz;
	
	cufftComplex *d_kvx_x, *d_kvx_z, *d_kvx_y, *d_kvz_x, *d_kvz_z, *d_kvz_y, *d_kvy_x, *d_kvy_z, *d_kvy_y;

	cufftComplex *d_partx1, *d_partz1, *d_party1, *d_partx2, *d_partz2, *d_party2, *d_partx3, *d_partz3, *d_party3;	

	cufftComplex *d_partvx_x1; cufftComplex *d_partvx_x2; cufftComplex *d_partvx_x3; cufftComplex *d_partvx_x4; cufftComplex *d_partvx_x5;
	cufftComplex *d_partvz_z1; cufftComplex *d_partvz_z2; cufftComplex *d_partvz_z3; cufftComplex *d_partvz_z4; cufftComplex *d_partvz_z5;
	cufftComplex *d_partvy_y1; cufftComplex *d_partvy_y2; cufftComplex *d_partvy_y3; cufftComplex *d_partvy_y4; cufftComplex *d_partvy_y5;
	cufftComplex *d_partvx_z1; cufftComplex *d_partvx_z2; cufftComplex *d_partvx_z3; cufftComplex *d_partvx_z4; cufftComplex *d_partvx_z5;
	cufftComplex *d_partvz_x1; cufftComplex *d_partvz_x2; cufftComplex *d_partvz_x3; cufftComplex *d_partvz_x4; cufftComplex *d_partvz_x5; 
	cufftComplex *d_partvy_z1; cufftComplex *d_partvy_z2; cufftComplex *d_partvy_z3; cufftComplex *d_partvy_z4; cufftComplex *d_partvy_z5;
	cufftComplex *d_partvz_y1; cufftComplex *d_partvz_y2; cufftComplex *d_partvz_y3; cufftComplex *d_partvz_y4; cufftComplex *d_partvz_y5; 
    cufftComplex *d_partvx_y1; cufftComplex *d_partvx_y2; cufftComplex *d_partvx_y3; cufftComplex *d_partvx_y4; cufftComplex *d_partvx_y5;
	cufftComplex *d_partvy_x1; cufftComplex *d_partvy_x2; cufftComplex *d_partvy_x3; cufftComplex *d_partvy_x4; cufftComplex *d_partvy_x5;

//	float *d_kfilter;

	float *d_eta_p1, *d_eta_s1, *d_eta_p2, *d_eta_s2, *d_eta_p3, *d_eta_s3, *d_tao_p1, *d_tao_s1, *d_tao_p2, *d_tao_s2;
	float *d_Ap1, *d_Ap2, *d_Ap3;

	float *d_record,*d_record2,*d_record3;
};

extern "C"
void cuda_forward_acoustic_3D
(
	int myid, int is, 
	int nt, int ntx, int nty, int ntz, int ntp, int nx, int ny, int nz, int pml,
	float dx, float dy, float dz, float dt, float f0, float w0, float velp_max, 
	float *rik, float *velp, float *gama_p, float *vels, float *gama_s, float *rho,
	struct Source ss[], struct MultiGPU plan[], int GPU_N, int rnmax, int rnx_max, int rny_max, int dr
);


extern "C"
void cuda_Host_initialization(int ntp,int nt, int rnmax, struct MultiGPU plan[], int GPU_N);

extern "C"
void cuda_Device_malloc
(
	int ntx, int nty, int ntz, int ntp, int nx, int ny, int nz, int nt,
	int rnmax,struct MultiGPU plan[], int GPU_N
);

extern "C"
void cuda_Device_free
(
	struct MultiGPU plan[], int GPU_N
);

void getrik(int nt, float dt, float f0, float *rik);
void get_velQ(int pml, int ntx, int nty, int ntz,int ntp, int nx, int ny, int nz, float *velp, float *Qp, float *vels, float *Qs, float *rho);
							
void get_homegeneous_velQ
(
	int is, int ntx, int nty, int ntz, float *velp, float *gama_p,
	float *c_velp, float *c_gama_p, int s_ix, int s_iy, int s_iz
);

		
