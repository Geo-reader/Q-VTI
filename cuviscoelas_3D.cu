#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "string.h"
#include "cuda.h"
#include "cufft.h"
#include "Myfunctions.h"
using namespace std;

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
//#define WITH_SHARED_MEMORY 0

#define pi 3.1415926

struct Multistream
{
	cudaStream_t stream,stream_back;
};

__global__ void cuda_kernel_wavenumber
(			
	int ntx, int nty, int ntz, float dx, float dy, float dz, float *kx, float *kz, float *ky, float *k,
    cufftComplex *kvx_x, cufftComplex *kvx_z, cufftComplex *kvx_y, cufftComplex *kvz_x, cufftComplex *kvz_z, cufftComplex *kvz_y,
    cufftComplex *kvy_x, cufftComplex *kvy_z, cufftComplex *kvy_y
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;

	int area=nty*ntz;
	int ip=ix*area+iy*ntz+iz;
    int ipt=iz*nty*ntx+ix*nty+iy;
    int iptt=iy*ntz*ntx+iz*ntx+ix;

	float dkx,dky,dkz;
    dkz=1.0/ntz/dz;
    dky=1.0/nty/dy;
    dkx=1.0/ntx/dx;

    float tmpx,tmpy,tmpz;
    tmpx=2*pi*dkx;
	tmpy=2*pi*dky;
    tmpz=2*pi*dkz;

	if(ix>=0 && ix<ntx && iy>=0 && iy< nty && iz>=0 && iz<ntz/2+1)
		kz[iz]=2*pi/ntz/dz*iz;
	if(ix>=0 && ix<ntx && iy>=0 && iy< nty && iz>=ntz/2+1 && iz<ntz)
		kz[iz]=2*pi/ntz/dz*(ntz-iz);
		
	if(iz>=0 && iz<ntz && iy>=0 && iy< nty && ix>=0 && ix<ntx/2+1)
		kx[ix]=2*pi/ntx/dx*ix;
	if(iz>=0 && iz<ntz && iy>=0 && iy< nty && ix>=ntx/2+1 && ix<ntx)
		kx[ix]=2*pi/ntx/dx*(ntx-ix);
		
	if(iz>=0 && iz<ntz && ix>=0 && ix<ntx && iy>=0 && iy< nty/2+1)
		ky[iy]=2*pi/nty/dy*iy;
	if(iz>=0 && iz<ntz && ix>=0 && ix<ntx && iy>=nty/2+1 && iy<nty)
		ky[iy]=2*pi/nty/dy*(nty-iy);		
		
		
	if(iz>=0 && iz<ntz && iy>=0 && iy< nty && ix>=0 && ix<ntx)
	{
		k[ip]=pow(kx[ix],2)+pow(kz[iz],2)+pow(ky[iy],2);
	}
	k[0]=1e-10;	


	if(ix>=0 && ix<ntx && iy>=0 && iy< nty && iz>=0 && iz<ntz/2+1)

		{
		kvz_z[ip].x=-tmpz*iz*sin(iz*pi/ntz);
		kvz_z[ip].y=tmpz*iz*cos(iz*pi/ntz);		
		
		kvx_z[ip].x=tmpz*iz*sin(iz*pi/ntz);
		kvx_z[ip].y=tmpz*iz*cos(iz*pi/ntz);	

		kvy_z[ip].x=tmpz*iz*sin(iz*pi/ntz);
		kvy_z[ip].y=tmpz*iz*cos(iz*pi/ntz);	
		}

	if(ix>=0 && ix<ntx && iy>=0 && iy< nty && iz>=ntz/2+1 && iz<ntz)
		{
		kvz_z[ip].x=-tmpz*(ntz-iz)*sin((ntz-iz)*pi/ntz);
		kvz_z[ip].y=-tmpz*(ntz-iz)*cos((ntz-iz)*pi/ntz);		
		
		kvx_z[ip].x=tmpz*(ntz-iz)*sin((ntz-iz)*pi/ntz);
		kvx_z[ip].y=-tmpz*(ntz-iz)*cos((ntz-iz)*pi/ntz);	

		kvy_z[ip].x=tmpz*(ntz-iz)*sin((ntz-iz)*pi/ntz);
		kvy_z[ip].y=-tmpz*(ntz-iz)*cos((ntz-iz)*pi/ntz);
		}	
		
	if(iz>=0 && iz<ntz && iy>=0 && iy< nty && ix>=0 && ix<ntx/2+1)
		{
		kvx_x[iptt].x=-tmpx*ix*sin(ix*pi/ntx);
		kvx_x[iptt].y=tmpx*ix*cos(ix*pi/ntx);
		
		kvz_x[iptt].x=tmpx*ix*sin(ix*pi/ntx);
		kvz_x[iptt].y=tmpx*ix*cos(ix*pi/ntx);  

		kvy_x[iptt].x=tmpx*ix*sin(ix*pi/ntx);
		kvy_x[iptt].y=tmpx*ix*cos(ix*pi/ntx); 
		}
 
	if(iz>=0 && iz<ntz && iy>=0 && iy< nty && ix>=ntx/2+1 && ix<ntx)
		{
		kvx_x[iptt].x=-tmpx*(ntx-ix)*sin((ntx-ix)*pi/ntx);
		kvx_x[iptt].y=-tmpx*(ntx-ix)*cos((ntx-ix)*pi/ntx);
		
		kvz_x[iptt].x=tmpx*(ntx-ix)*sin((ntx-ix)*pi/ntx);
		kvz_x[iptt].y=-tmpx*(ntx-ix)*cos((ntx-ix)*pi/ntx);  

		kvy_x[iptt].x=tmpx*(ntx-ix)*sin((ntx-ix)*pi/ntx);
		kvy_x[iptt].y=-tmpx*(ntx-ix)*cos((ntx-ix)*pi/ntx); 
		}
		
		
	if(iz>=0 && iz<ntz && ix>=0 && ix<ntx && iy>=0 && iy< nty/2+1)
		{
		kvy_y[ipt].x=-tmpy*iy*sin(iy*pi/nty);
		kvy_y[ipt].y=tmpy*iy*cos(iy*pi/nty);

		kvz_y[ipt].x=tmpy*iy*sin(iy*pi/nty);
		kvz_y[ipt].y=tmpy*iy*cos(iy*pi/nty);

		kvx_y[ipt].x=tmpy*iy*sin(iy*pi/nty);
		kvx_y[ipt].y=tmpy*iy*cos(iy*pi/nty);
		}

	if(iz>=0 && iz<ntz && ix>=0 && ix<ntx && iy>=nty/2+1 && iy<nty)
		{
		kvy_y[ipt].x=-tmpy*(nty-iy)*sin((nty-iy)*pi/nty);
		kvy_y[ipt].y=-tmpy*(nty-iy)*cos((nty-iy)*pi/nty);

		kvz_y[ipt].x=tmpy*(nty-iy)*sin((nty-iy)*pi/nty);
		kvz_y[ipt].y=-tmpy*(nty-iy)*cos((nty-iy)*pi/nty);

		kvx_y[ipt].x=tmpy*(nty-iy)*sin((nty-iy)*pi/nty);
		kvx_y[ipt].y=-tmpy*(nty-iy)*cos((nty-iy)*pi/nty);
		}
	
	__syncthreads();
}


__global__ void cuda_kernel_viscoacoustic_parameters
(
	int ntx, int nty, int ntz, float dx, float dy, float dz, float dt, float w0,
	float *velp, float *vels, float *rho, float *k, float *gama_p, float *gama_s,
    float *Ap1, float *Ap2, float *Ap3, 
	float *tao_p1, float *tao_p2, float *eta_p1, float *eta_p2, float *eta_p3, 
	float *tao_s1, float *tao_s2, float *eta_s1, float *eta_s2, float *eta_s3
)

{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;

	int area=nty*ntz;
	int ip=ix*area+iy*ntz+iz;
	
	float sinc2nd;
	float vel;

	if(iz>=0 && iz<ntz && iy>=0 && iy< nty && ix>=0 && ix<ntx)
	{
//		sinc2nd =1.0; //pow(sin(velp_max*powf(k[ip],0.5)*dt/2)/(velp_max*powf(k[ip],0.5)*dt/2),2);
		sinc2nd =1.0; //pow(sin(vel*powf(k[ip],0.5)*dt/2)/(vel*powf(k[ip],0.5)*dt/2),2);		
		

//////////////////////////////
		Ap1[ip]=sinc2nd*powf(k[ip],-0.5);
        Ap2[ip]=sinc2nd;
        Ap3[ip]=sinc2nd*powf(k[ip],0.5);
/////////////////////////////	
//////////////////////////////////


		tao_p1[ip]=rho[ip]*pow(velp[ip]*cos(gama_p[ip]*pi/2.0),1)*gama_p[ip]*pi;
        tao_s1[ip]=rho[ip]*pow(vels[ip]*cos(gama_s[ip]*pi/2.0),1)*gama_s[ip]*pi;
		tao_p2[ip]=rho[ip]*pow(velp[ip]*cos(gama_p[ip]*pi/2.0),2)*pow(gama_p[ip],2)*pi/w0;
        tao_s2[ip]=rho[ip]*pow(vels[ip]*cos(gama_s[ip]*pi/2.0),2)*pow(gama_s[ip],2)*pi/w0;

		eta_p1[ip]=-rho[ip]*pow(velp[ip]*cos(gama_p[ip]*pi/2.0),1)*gama_p[ip]*w0;						
		eta_s1[ip]=-rho[ip]*pow(vels[ip]*cos(gama_s[ip]*pi/2.0),1)*gama_s[ip]*w0;	
		eta_p2[ip]=rho[ip]*pow(velp[ip]*cos(gama_p[ip]*pi/2.0),2);							
		eta_s2[ip]=rho[ip]*pow(vels[ip]*cos(gama_s[ip]*pi/2.0),2);	
		eta_p3[ip]=rho[ip]*pow(velp[ip]*cos(gama_p[ip]*pi/2.0),3)*gama_p[ip]/w0;						
		eta_s3[ip]=rho[ip]*pow(vels[ip]*cos(gama_s[ip]*pi/2.0),3)*gama_s[ip]/w0;
/////////////////////////////////					
	}
	
	__syncthreads();
}


__global__ void cuda_kernel_pml_parameters
(
	int ntx, int nty, int ntz, int pml, float dx, float dy, float dz, float dt, float f0, float velp_max,
	float *gammax, float *alphax, float *Omegax, float *a_x, float *b_x,
	float *gammay, float *alphay, float *Omegay, float *a_y, float *b_y,
	float *gammaz, float *alphaz, float *Omegaz, float *a_z, float *b_z	
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;
	
	int n1=2;
	int n2=1; 
	int n3=2;
	float R=1e-2;
    velp_max=5000;
	
//	float gamma_max = 1.0;	
	float alpha_max = pi*f0;
	float Omegax_max = (1+n1+n2)*velp_max*log(1.0/R)/((n1+n2-1)*pml*dx);
	float Omegay_max = (1+n1+n2)*velp_max*log(1.0/R)/((n1+n2-1)*pml*dy);
	float Omegaz_max = (1+n1+n2)*velp_max*log(1.0/R)/((n1+n2-1)*pml*dz);
        
	if(ix>=0&&ix<=pml-1)
	{
		gammax[ix] = 1.0;// + (gamma_max-1)*powf(1.0*ix/(pml-1),n1);
		alphax[ix] = alpha_max*powf(1.0*ix/(pml-1),n3);
		Omegax[ix] = Omegax_max*powf(1.0*(pml-1-ix)/pml,n1+n2);
		gammax[ntx-1-ix] = gammax[ix];
		alphax[ntx-1-ix] = alphax[ix];
		Omegax[ntx-1-ix] = Omegax[ix];
	}
	if(ix>=pml&&ix<=ntx-1-pml)
	{
		gammax[ix] = 1.0;
		alphax[ix] = alpha_max;
		Omegax[ix] = 0.0;
	}

	if(iy>=0&&iy<=pml-1)
	{
		gammay[iy] = 1.0;// + (gamma_max-1)*powf(1.0*ix/(pml-1),n1);
		alphay[iy] = alpha_max*powf(1.0*iy/(pml-1),n3);
		Omegay[iy] = Omegay_max*powf(1.0*(pml-1-iy)/pml,n1+n2);
		gammay[nty-1-iy] = gammay[iy];
		alphay[nty-1-iy] = alphay[iy];
		Omegay[nty-1-iy] = Omegay[iy];
	}
	if(iy>=pml&&iy<=nty-1-pml)
	{
		gammay[iy] = 1.0;
		alphay[iy] = alpha_max;
		Omegay[iy] = 0.0;
	}

	if(iz>=0&&iz<=pml-1)
	{
		gammaz[iz] = 1.0;// + (gamma_max-1)*gamma_max*powf(1.0*iz/(pml-1),n1);
		alphaz[iz] = alpha_max*powf(1.0*iz/(pml-1),n3);
		Omegaz[iz] = Omegaz_max*powf(1.0*(pml-1-iz)/pml,n1+n2);
		gammaz[ntz-1-iz] = gammaz[iz];
		alphaz[ntz-1-iz] = alphaz[iz];
		Omegaz[ntz-1-iz] = Omegaz[iz];
	}
	if(iz>=pml&&iz<=ntz-1-pml)
	{
		gammaz[iz] = 1.0;
		alphaz[iz] = alpha_max;
		Omegaz[iz] = 0.0;
	}

	if(ix>=0&&ix<=ntx-1)
	{
		a_x[ix] = alphax[ix] + Omegax[ix]/gammax[ix];
		b_x[ix] = Omegax[ix]/powf(gammax[ix],2.0);
	}

	if(iy>=0&&iy<=nty-1)
	{
		a_y[iy] = alphay[iy] + Omegay[iy]/gammay[iy];
		b_y[iy] = Omegay[iy]/powf(gammay[iy],2.0);
	}

	if(iz>=0&&iz<=ntz-1)
	{
		a_z[iz] = alphaz[iz] + Omegaz[iz]/gammaz[iz];
		b_z[iz] = Omegaz[iz]/powf(gammaz[iz],2.0);
	}
	__syncthreads();
}


__global__ void cuda_kernel_initialization
(
	int ntx, int nty, int ntz,
	float *vx, float *vy, float *vz, 
	float *pxx, float *pyy, float *pzz,
	float *pxy, float *pyz, float *pxz,
	float *phi_vx_xx, float *phi_vz_zx, float *phi_vy_yx, 
	float *phi_vx_xy, float *phi_vz_zy, float *phi_vy_yy, 
	float *phi_vx_xz, float *phi_vz_zz, float *phi_vy_yz, 
	float *phi_vx_z,  float *phi_vz_x,  float *phi_vx_y, 
	float *phi_vy_x,  float *phi_vy_z,  float *phi_vz_y,
	float *phi_pxx_x, float *phi_pxy_y, float *phi_pxz_z,
	float *phi_pxy_x, float *phi_pyy_y, float *phi_pyz_z, 
	float *phi_pxz_x, float *phi_pyz_y, float *phi_pzz_z,
	cufftComplex *dvx, cufftComplex *dvy, cufftComplex *dvz,
	cufftComplex *partx1, cufftComplex *partz1, cufftComplex *party1,
	cufftComplex *partx2, cufftComplex *partz2, cufftComplex *party2,
	cufftComplex *partx3, cufftComplex *partz3, cufftComplex *party3,
	cufftComplex *partvx_x1, cufftComplex *partvx_x2, cufftComplex *partvx_x3, cufftComplex *partvx_x4, cufftComplex *partvx_x5,
	cufftComplex *partvz_z1, cufftComplex *partvz_z2, cufftComplex *partvz_z3, cufftComplex *partvz_z4, cufftComplex *partvz_z5,
	cufftComplex *partvy_y1, cufftComplex *partvy_y2, cufftComplex *partvy_y3, cufftComplex *partvy_y4, cufftComplex *partvy_y5,
	cufftComplex *partvx_y1, cufftComplex *partvx_y2, cufftComplex *partvx_y3, cufftComplex *partvx_y4, cufftComplex *partvx_y5,
	cufftComplex *partvy_x1, cufftComplex *partvy_x2, cufftComplex *partvy_x3, cufftComplex *partvy_x4, cufftComplex *partvy_x5,
	cufftComplex *partvy_z1, cufftComplex *partvy_z2, cufftComplex *partvy_z3, cufftComplex *partvy_z4, cufftComplex *partvy_z5,
	cufftComplex *partvz_y1, cufftComplex *partvz_y2, cufftComplex *partvz_y3, cufftComplex *partvz_y4, cufftComplex *partvz_y5,
	cufftComplex *partvx_z1, cufftComplex *partvx_z2, cufftComplex *partvx_z3, cufftComplex *partvx_z4, cufftComplex *partvx_z5,
	cufftComplex *partvz_x1, cufftComplex *partvz_x2, cufftComplex *partvz_x3, cufftComplex *partvz_x4, cufftComplex *partvz_x5
)

{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;

	int area=nty*ntz;
	int ip=ix*area+iy*ntz+iz;


	if(iz>=0 && iz<ntz && iy>=0 && iy< nty && ix>=0 && ix<ntx)
	{
		vx[ip]=0.0;vy[ip]=0.0;vz[ip]=0.0;		
		pxx[ip]=0.0; pyy[ip]=0.0; pzz[ip]=0.0;
		pxy[ip]=0.0; pyz[ip]=0.0; pxz[ip]=0.0;

		phi_vx_xx[ip]=0.0; phi_vz_zx[ip]=0.0; phi_vy_yx[ip]=0.0;
		phi_vx_xy[ip]=0.0; phi_vz_zy[ip]=0.0; phi_vy_yy[ip]=0.0;
		phi_vx_xz[ip]=0.0; phi_vz_zz[ip]=0.0; phi_vy_yz[ip]=0.0;		
		phi_vx_z[ip]=0.0; phi_vz_x[ip]=0.0;
		phi_vx_y[ip]=0.0; phi_vy_x[ip]=0.0;
		phi_vy_z[ip]=0.0; phi_vz_y[ip]=0.0;
		phi_pxx_x[ip]=0.0; phi_pxy_y[ip]=0.0; phi_pxz_z[ip]=0.0;
		phi_pxy_x[ip]=0.0; phi_pyy_y[ip]=0.0; phi_pyz_z[ip]=0.0;
		phi_pxz_x[ip]=0.0; phi_pyz_y[ip]=0.0; phi_pzz_z[ip]=0.0;			
		
		partx1[ip].x=0.0;	 partx1[ip].y=0.0; party1[ip].x=0.0;	 party1[ip].y=0.0; partz1[ip].x=0.0;	 partz1[ip].y=0.0;
		partx2[ip].x=0.0;	 partx2[ip].y=0.0; party2[ip].x=0.0;	 party2[ip].y=0.0; partz2[ip].x=0.0;	 partz2[ip].y=0.0;
		partx3[ip].x=0.0;	 partx3[ip].y=0.0; party3[ip].x=0.0;	 party3[ip].y=0.0; partz3[ip].x=0.0;	 partz3[ip].y=0.0;
		
		partvx_x1[ip].x=0.0; partvx_x1[ip].y=0.0; partvz_z1[ip].x=0.0; partvz_z1[ip].y=0.0; partvy_y1[ip].x=0.0; partvy_y1[ip].y=0.0;
        partvx_x2[ip].x=0.0; partvx_x2[ip].y=0.0; partvz_z2[ip].x=0.0; partvz_z2[ip].y=0.0; partvy_y2[ip].x=0.0; partvy_y2[ip].y=0.0;
		partvx_x3[ip].x=0.0; partvx_x3[ip].y=0.0; partvz_z3[ip].x=0.0; partvz_z3[ip].y=0.0; partvy_y3[ip].x=0.0; partvy_y3[ip].y=0.0;
		partvx_x4[ip].x=0.0; partvx_x4[ip].y=0.0; partvz_z4[ip].x=0.0; partvz_z4[ip].y=0.0; partvy_y4[ip].x=0.0; partvy_y4[ip].y=0.0;
		partvx_x5[ip].x=0.0; partvx_x5[ip].y=0.0; partvz_z5[ip].x=0.0; partvz_z5[ip].y=0.0; partvy_y5[ip].x=0.0; partvy_y5[ip].y=0.0;


		partvx_y1[ip].x=0.0; partvx_y1[ip].y=0.0; partvy_x1[ip].x=0.0; partvy_x1[ip].y=0.0;
        partvx_y2[ip].x=0.0; partvx_y2[ip].y=0.0; partvy_x2[ip].x=0.0; partvy_x2[ip].y=0.0;
		partvx_y3[ip].x=0.0; partvx_y3[ip].y=0.0; partvy_x3[ip].x=0.0; partvy_x3[ip].y=0.0;
		partvx_y4[ip].x=0.0; partvx_y4[ip].y=0.0; partvy_x4[ip].x=0.0; partvy_x4[ip].y=0.0;
		partvx_y5[ip].x=0.0; partvx_y5[ip].y=0.0; partvy_x5[ip].x=0.0; partvy_x5[ip].y=0.0;

	
		partvy_z1[ip].x=0.0; partvy_z1[ip].y=0.0; partvz_y1[ip].x=0.0; partvz_y1[ip].y=0.0;
        partvy_z2[ip].x=0.0; partvy_z2[ip].y=0.0; partvz_y2[ip].x=0.0; partvz_y2[ip].y=0.0;
		partvy_z3[ip].x=0.0; partvy_z3[ip].y=0.0; partvz_y3[ip].x=0.0; partvz_y3[ip].y=0.0;
		partvy_z4[ip].x=0.0; partvy_z4[ip].y=0.0; partvz_y4[ip].x=0.0; partvz_y4[ip].y=0.0;
		partvy_z5[ip].x=0.0; partvy_z5[ip].y=0.0; partvz_y5[ip].x=0.0; partvz_y5[ip].y=0.0;


		partvz_x1[ip].x=0.0; partvz_x1[ip].y=0.0; partvx_z1[ip].x=0.0; partvx_z1[ip].y=0.0;
        partvz_x2[ip].x=0.0; partvz_x2[ip].y=0.0; partvx_z2[ip].x=0.0; partvx_z2[ip].y=0.0;
		partvz_x3[ip].x=0.0; partvz_x3[ip].y=0.0; partvx_z3[ip].x=0.0; partvx_z3[ip].y=0.0;
		partvz_x4[ip].x=0.0; partvz_x4[ip].y=0.0; partvx_z4[ip].x=0.0; partvx_z4[ip].y=0.0;
		partvz_x5[ip].x=0.0; partvz_x5[ip].y=0.0; partvx_z5[ip].x=0.0; partvx_z5[ip].y=0.0;


		dvx[ip].x=0.0;			dvx[ip].y=0.0;
		dvy[ip].x=0.0;			dvy[ip].y=0.0;
		dvz[ip].x=0.0;			dvz[ip].y=0.0;
	}
	
	__syncthreads();	
}


__global__ void cuda_kernel_p_real_to_complex
(
	int ntx, int nty, int ntz, 
	float *real_pxx, float *real_pyy, float *real_pzz, float *real_pxy, float *real_pxz, float *real_pyz,
	cufftComplex *in_pxx, cufftComplex *in_pyy, cufftComplex *in_pzz, cufftComplex *in_pxy, cufftComplex *in_pxz, cufftComplex *in_pyz
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;

	int area=nty*ntz;
	int ip=ix*area+iy*ntz+iz;

	if(iz>=0 && iz<ntz && iy>=0 && iy< nty && ix>=0 && ix<ntx)
	{
		in_pxx[ip].x=real_pxx[ip]; 	in_pxx[ip].y=0.0;	
		in_pyy[ip].x=real_pyy[ip]; 	in_pyy[ip].y=0.0;	
		in_pzz[ip].x=real_pzz[ip]; 	in_pzz[ip].y=0.0;	
		in_pxy[ip].x=real_pxy[ip]; 	in_pxy[ip].y=0.0;	
		in_pyz[ip].x=real_pyz[ip]; 	in_pyz[ip].y=0.0;	
		in_pxz[ip].x=real_pxz[ip]; 	in_pxz[ip].y=0.0;
	}
	
	__syncthreads();	
}

__global__ void cuda_kernel_vxvyvz_real_to_complex
(
	int ntx, int nty, int ntz, float *real_x, float *real_y, float *real_z, cufftComplex *inx, cufftComplex *iny, cufftComplex *inz
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;

	int area=nty*ntz;
	int ip=ix*area+iy*ntz+iz;

	if(iz>=0 && iz<ntz && iy>=0 && iy< nty && ix>=0 && ix<ntx)
	{
		inx[ip].x=real_x[ip]; 			inx[ip].y=0.0;	
		iny[ip].x=real_y[ip]; 			iny[ip].y=0.0;	
		inz[ip].x=real_z[ip]; 			inz[ip].y=0.0;	
	}
	
	__syncthreads();	
}

			
__global__ void cuda_kernel_operate_k_pxxpyypzz
(
	int ntx, int nty, int ntz, float dt, 
	cufftComplex *outx, cufftComplex *outy, cufftComplex *outz, cufftComplex *dvx, cufftComplex *dvy, cufftComplex *dvz,
	cufftComplex *inx, cufftComplex *iny, cufftComplex *inz, cufftComplex *k_x, cufftComplex *k_y, cufftComplex *k_z, float *k2, int AorB
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;

	int area=nty*ntz;
	int ip=ix*area+iy*ntz+iz;
	int ipt=iz*nty*ntx+ix*nty+iy;
    int iptt=iy*ntz*ntx+iz*ntx+ix;

	cufftComplex tmpx, tmpy, tmpz;
	
	if(iz>=0 && iz<ntz && ix>=0 && ix<ntx && iy>=0 && iy< nty)
	{
		if(AorB==0)
		{

			inx[ip].x=k2[ip]*(k_x[iptt].x*outx[ip].x - k_x[iptt].y*outx[ip].y);
			inx[ip].y=k2[ip]*(k_x[iptt].x*outx[ip].y + k_x[iptt].y*outx[ip].x);		

			iny[ip].x=k2[ip]*(k_y[ipt].x*outy[ip].x - k_y[ipt].y*outy[ip].y);
			iny[ip].y=k2[ip]*(k_y[ipt].x*outy[ip].y + k_y[ipt].y*outy[ip].x);
		
			inz[ip].x=k2[ip]*(k_z[ip].x*outz[ip].x - k_z[ip].y*outz[ip].y);
			inz[ip].y=k2[ip]*(k_z[ip].x*outz[ip].y + k_z[ip].y*outz[ip].x);
		}

		if(AorB==1)
		{

				tmpx.x=(outx[ip].x-dvx[ip].x)/dt;
				tmpx.y=(outx[ip].y-dvx[ip].y)/dt;
				tmpy.x=(outy[ip].x-dvy[ip].x)/dt;
				tmpy.y=(outy[ip].y-dvy[ip].y)/dt;
				tmpz.x=(outz[ip].x-dvz[ip].x)/dt;
				tmpz.y=(outz[ip].y-dvz[ip].y)/dt;


				inx[ip].x=k2[ip]*(k_x[iptt].x*tmpx.x - k_x[iptt].y*tmpx.y);
				inx[ip].y=k2[ip]*(k_x[iptt].x*tmpx.y + k_x[iptt].y*tmpx.x);	
				iny[ip].x=k2[ip]*(k_y[ipt].x*tmpy.x - k_y[ipt].y*tmpy.y);
				iny[ip].y=k2[ip]*(k_y[ipt].x*tmpy.y + k_y[ipt].y*tmpy.x);	
				inz[ip].x=k2[ip]*(k_z[ip].x*tmpz.x - k_z[ip].y*tmpz.y);
				inz[ip].y=k2[ip]*(k_z[ip].x*tmpz.y + k_z[ip].y*tmpz.x);					
		}
	}
	
	__syncthreads();	
}


__global__ void cuda_kernel_operate_k_pxz
(
	int ntx, int nty, int ntz, float dt, 
	cufftComplex *outx, cufftComplex *outz, cufftComplex *dvx, cufftComplex *dvz,
	cufftComplex *inx, cufftComplex *inz, cufftComplex *k_x, cufftComplex *k_z, float *k2, int AorB
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;

	int area=nty*ntz;
	int ip=ix*area+iy*ntz+iz;
	int ipt=iz*nty*ntx+ix*nty+iy;
    int iptt=iy*ntz*ntx+iz*ntx+ix;

	cufftComplex tmpx, tmpy, tmpz;
	
	if(iz>=0 && iz<ntz && ix>=0 && ix<ntx && iy>=0 && iy< nty)
	{
		if(AorB==0)
		{

			inx[ip].x=k2[ip]*(k_x[ip].x*outx[ip].x - k_x[ip].y*outx[ip].y);
			inx[ip].y=k2[ip]*(k_x[ip].x*outx[ip].y + k_x[ip].y*outx[ip].x);		

		
			inz[ip].x=k2[ip]*(k_z[iptt].x*outz[ip].x - k_z[iptt].y*outz[ip].y);
			inz[ip].y=k2[ip]*(k_z[iptt].x*outz[ip].y + k_z[iptt].y*outz[ip].x);
		}

		if(AorB==1)
		{

				tmpx.x=(outx[ip].x-dvx[ip].x)/dt;
				tmpx.y=(outx[ip].y-dvx[ip].y)/dt;
				tmpz.x=(outz[ip].x-dvz[ip].x)/dt;
				tmpz.y=(outz[ip].y-dvz[ip].y)/dt;


				inx[ip].x=k2[ip]*(k_x[ip].x*tmpx.x - k_x[ip].y*tmpx.y);
				inx[ip].y=k2[ip]*(k_x[ip].x*tmpx.y + k_x[ip].y*tmpx.x);	
	
				inz[ip].x=k2[ip]*(k_z[iptt].x*tmpz.x - k_z[iptt].y*tmpz.y);
				inz[ip].y=k2[ip]*(k_z[iptt].x*tmpz.y + k_z[iptt].y*tmpz.x);					
		}
	}
	
	__syncthreads();	
}


__global__ void cuda_kernel_operate_k_pxy
(
	int ntx, int nty, int ntz, float dt, 
	cufftComplex *outx, cufftComplex *outy, cufftComplex *dvx, cufftComplex *dvy,
	cufftComplex *inx, cufftComplex *iny, cufftComplex *k_x, cufftComplex *k_y, float *k2, int AorB
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;

	int area=nty*ntz;
	int ip=ix*area+iy*ntz+iz;
	int ipt=iz*nty*ntx+ix*nty+iy;
    int iptt=iy*ntz*ntx+iz*ntx+ix;

	cufftComplex tmpx, tmpy, tmpz;
	
	if(iz>=0 && iz<ntz && ix>=0 && ix<ntx && iy>=0 && iy< nty)
	{
		if(AorB==0)
		{

			inx[ip].x=k2[ip]*(k_x[ipt].x*outx[ip].x - k_x[ipt].y*outx[ip].y);
			inx[ip].y=k2[ip]*(k_x[ipt].x*outx[ip].y + k_x[ipt].y*outx[ip].x);		

			iny[ip].x=k2[ip]*(k_y[iptt].x*outy[ip].x - k_y[iptt].y*outy[ip].y);
			iny[ip].y=k2[ip]*(k_y[iptt].x*outy[ip].y + k_y[iptt].y*outy[ip].x);
		}

		if(AorB==1)
		{

				tmpx.x=(outx[ip].x-dvx[ip].x)/dt;
				tmpx.y=(outx[ip].y-dvx[ip].y)/dt;
				tmpy.x=(outy[ip].x-dvy[ip].x)/dt;
				tmpy.y=(outy[ip].y-dvy[ip].y)/dt;


				inx[ip].x=k2[ip]*(k_x[ipt].x*tmpx.x - k_x[ipt].y*tmpx.y);
				inx[ip].y=k2[ip]*(k_x[ipt].x*tmpx.y + k_x[ipt].y*tmpx.x);	
				iny[ip].x=k2[ip]*(k_y[iptt].x*tmpy.x - k_y[iptt].y*tmpy.y);
				iny[ip].y=k2[ip]*(k_y[iptt].x*tmpy.y + k_y[iptt].y*tmpy.x);					
		}
	}
	
	__syncthreads();	
}

__global__ void cuda_kernel_operate_k_pyz
(
	int ntx, int nty, int ntz, float dt, 
	cufftComplex *outy, cufftComplex *outz, cufftComplex *dvy, cufftComplex *dvz,
	cufftComplex *iny, cufftComplex *inz, cufftComplex *k_y, cufftComplex *k_z, float *k2, int AorB
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;

	int area=nty*ntz;
	int ip=ix*area+iy*ntz+iz;
	int ipt=iz*nty*ntx+ix*nty+iy;
    int iptt=iy*ntz*ntx+iz*ntx+ix;

	cufftComplex tmpx, tmpy, tmpz;
	
	if(iz>=0 && iz<ntz && ix>=0 && ix<ntx && iy>=0 && iy< nty)
	{
		if(AorB==0)
		{	

			iny[ip].x=k2[ip]*(k_y[ip].x*outy[ip].x - k_y[ip].y*outy[ip].y);
			iny[ip].y=k2[ip]*(k_y[ip].x*outy[ip].y + k_y[ip].y*outy[ip].x);
		
			inz[ip].x=k2[ip]*(k_z[ipt].x*outz[ip].x - k_z[ipt].y*outz[ip].y);
			inz[ip].y=k2[ip]*(k_z[ipt].x*outz[ip].y + k_z[ipt].y*outz[ip].x);
		}

		if(AorB==1)
		{

				tmpy.x=(outy[ip].x-dvy[ip].x)/dt;
				tmpy.y=(outy[ip].y-dvy[ip].y)/dt;
				tmpz.x=(outz[ip].x-dvz[ip].x)/dt;
				tmpz.y=(outz[ip].y-dvz[ip].y)/dt;


				iny[ip].x=k2[ip]*(k_y[ip].x*tmpy.x - k_y[ip].y*tmpy.y);
				iny[ip].y=k2[ip]*(k_y[ip].x*tmpy.y + k_y[ip].y*tmpy.x);	
				inz[ip].x=k2[ip]*(k_z[ipt].x*tmpz.x - k_z[ipt].y*tmpz.y);
				inz[ip].y=k2[ip]*(k_z[ipt].x*tmpz.y + k_z[ipt].y*tmpz.x);					
		}
	}
	
	__syncthreads();	
}


__global__ void cuda_kernel_operate_k_v
(
	int ntx, int nty, int ntz, float dt, cufftComplex *outx,  cufftComplex *outy,  cufftComplex *outz,  
	cufftComplex *inx, cufftComplex *iny, cufftComplex *inz, cufftComplex *k_x, cufftComplex *k_y, cufftComplex *k_z
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;

	int area=nty*ntz;
	int ip=ix*area+iy*ntz+iz;
	int ipt=iz*nty*ntx+ix*nty+iy;
    int iptt=iy*ntz*ntx+iz*ntx+ix;

	cufftComplex tmpx, tmpy, tmpz;
	
	if(iz>=0 && iz<ntz && ix>=0 && ix<ntx && iy>=0 && iy< nty)
	{

			inx[ip].x=k_x[iptt].x*outx[ip].x - k_x[iptt].y*outx[ip].y;
			inx[ip].y=k_x[iptt].x*outx[ip].y + k_x[iptt].y*outx[ip].x;	

			iny[ip].x=k_y[ipt].x*outy[ip].x - k_y[ipt].y*outy[ip].y;
			iny[ip].y=k_y[ipt].x*outy[ip].y + k_y[ipt].y*outy[ip].x;	
		
			inz[ip].x=k_z[ip].x*outz[ip].x - k_z[ip].y*outz[ip].y;
			inz[ip].y=k_z[ip].x*outz[ip].y + k_z[ip].y*outz[ip].x;
	}
	
	__syncthreads();	
}


__global__ void cuda_kernel_forward_IO
(
	int ntx, int nty, int ntz, int ntp, int pml, int nt, int it, float dx, float dy, float dz, float dt, 
	int s_ix, int s_iy, int s_iz, float *rik, 
	float *record, float *record2, float *record3, int *r_ix, int *r_iy, int r_iz, int rnmax, int rnx_max, int rny_max, int dr, int r_n,
	float *pxx, float *pyy, float *pzz, float *vx, float *vy, float *vz
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;

	int area=nty*ntz;
	int ip=ix*area+iy*ntz+iz;
    int ip11=(ix+15)*area+(iy+15)*ntz+iz;
	
	int ir;

	//============Add source==============//
		if(iz==s_iz+10 && ix==s_ix && iy==s_iy)
		{
			vx[ip]+=rik[it];
			//pyy[ip]+=rik[it];
			//pzz[ip]+=rik[it];
		}


	//===============seismic record=================//	

		if(ix>=0&&ix<rnx_max && iy>=0&&iy<rny_max&& iz==r_iz)
		{
		/*	ir=ix*rny_max+iy;
//			record[it*rnmax+(r_iy[ir]-r_iy[0])/dr*rnx_max+(r_ix[ir]-r_ix[0])/dr]=p2[iz*area+r_iy[ir]*ntx+r_ix[ir]];
			record[ir*nt+it]=vx[r_ix[ir]*area+r_iy[ir]*ntz+iz];*/
			ir=ix*rny_max+iy;
//			record[it*rnmax+(r_iy[ir]-r_iy[0])/dr*rnx_max+(r_ix[ir]-r_ix[0])/dr]=p2[iz*area+r_iy[ir]*ntx+r_ix[ir]];
			record[it*rny_max*rnx_max+ir]=vx[r_ix[ir]*area+r_iy[ir]*ntz+iz];
			record2[it*rny_max*rnx_max+ir]=vy[r_ix[ir]*area+r_iy[ir]*ntz+iz];
			record3[it*rny_max*rnx_max+ir]=vz[r_ix[ir]*area+r_iy[ir]*ntz+iz];

		}

		
	__syncthreads();	
}


__global__ void cuda_kernel_calculate_p

(
	int ntx, int nty, int ntz, int ntp, float dt, 
	float *pxx, float *pyy, float *pzz, float *pxy, float *pxz, float *pyz, 
    float *tao_p1, float *tao_s1, float *tao_p2, float *tao_s2, float *eta_p1, float *eta_s1, float *eta_p2, float *eta_s2, float *eta_p3, float *eta_s3,   
	float *gammax, float *a_x, float *b_x,
	float *gammay, float *a_y, float *b_y,
	float *gammaz, float *a_z, float *b_z,
	float *phi_vx_xx, float *phi_vz_zx, float *phi_vy_yx, 
	float *phi_vx_xy, float *phi_vz_zy, float *phi_vy_yy, 
	float *phi_vx_xz, float *phi_vz_zz, float *phi_vy_yz, 
	float *phi_vx_z,  float *phi_vz_x,  float *phi_vx_y, 
	float *phi_vy_x,  float *phi_vy_z,  float *phi_vz_y,
	cufftComplex *partvx_x1, cufftComplex *partvx_x2, cufftComplex *partvx_x3, cufftComplex *partvx_x4, cufftComplex *partvx_x5,
	cufftComplex *partvz_z1, cufftComplex *partvz_z2, cufftComplex *partvz_z3, cufftComplex *partvz_z4, cufftComplex *partvz_z5,
	cufftComplex *partvy_y1, cufftComplex *partvy_y2, cufftComplex *partvy_y3, cufftComplex *partvy_y4, cufftComplex *partvy_y5,
	cufftComplex *partvx_z1, cufftComplex *partvx_z2, cufftComplex *partvx_z3, cufftComplex *partvx_z4, cufftComplex *partvx_z5,
	cufftComplex *partvz_x1, cufftComplex *partvz_x2, cufftComplex *partvz_x3, cufftComplex *partvz_x4, cufftComplex *partvz_x5,
	cufftComplex *partvx_y1, cufftComplex *partvx_y2, cufftComplex *partvx_y3, cufftComplex *partvx_y4, cufftComplex *partvx_y5,
	cufftComplex *partvy_x1, cufftComplex *partvy_x2, cufftComplex *partvy_x3, cufftComplex *partvy_x4, cufftComplex *partvy_x5,
	cufftComplex *partvy_z1, cufftComplex *partvy_z2, cufftComplex *partvy_z3, cufftComplex *partvy_z4, cufftComplex *partvy_z5,
	cufftComplex *partvz_y1, cufftComplex *partvz_y2, cufftComplex *partvz_y3, cufftComplex *partvz_y4, cufftComplex *partvz_y5
)

{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;

	int area=nty*ntz;
	int ip=ix*area+iy*ntz+iz;
	
	float alpha=1.0;
	float w, s, t11, t12, t13;
	float sign_of_tao;

	if(iz>=0 && iz<ntz-1 && ix>=0 && ix<ntx-1 && iy>=0 && iy<nty-1)
	{

        phi_vx_xx[ip] = phi_vx_xx[ip] + dt*(-a_x[ix]*phi_vx_xx[ip] - b_x[ix]*(
										eta_p1[ip]*partvx_x1[ip].x/ntp+eta_p2[ip]*partvx_x2[ip].x/ntp+eta_p3[ip]*partvx_x3[ip].x/ntp+tao_p1[ip]*partvx_x4[ip].x/ntp+tao_p2[ip]*partvx_x5[ip].x/ntp));


		phi_vz_zx[ip] = phi_vz_zx[ip] + dt*(-a_z[iz]*phi_vz_zx[ip] - b_z[iz]*(
										(eta_p1[ip]-2*eta_s1[ip])*partvz_z1[ip].x/ntp + (eta_p2[ip]-2*eta_s2[ip])*partvz_z2[ip].x/ntp + (eta_p3[ip]-2*eta_s3[ip])*partvz_z3[ip].x/ntp
														+(tao_p1[ip]-2*tao_s1[ip])*partvz_z4[ip].x/ntp + (tao_p2[ip]-2*tao_s2[ip])*partvz_z5[ip].x/ntp));

		phi_vy_yx[ip] = phi_vy_yx[ip] + dt*(-a_y[iy]*phi_vy_yx[ip] - b_y[iy]*(
										(eta_p1[ip]-2*eta_s1[ip])*partvy_y1[ip].x/ntp + (eta_p2[ip]-2*eta_s2[ip])*partvy_y2[ip].x/ntp + (eta_p3[ip]-2*eta_s3[ip])*partvy_y3[ip].x/ntp
														+(tao_p1[ip]-2*tao_s1[ip])*partvy_y4[ip].x/ntp + (tao_p2[ip]-2*tao_s2[ip])*partvy_y5[ip].x/ntp));
        		
        phi_vz_zz[ip] = phi_vz_zz[ip] + dt*(-a_z[iz]*phi_vz_zz[ip] - b_z[iz]*(
										eta_p1[ip]*partvz_z1[ip].x/ntp+eta_p2[ip]*partvz_z2[ip].x/ntp+eta_p3[ip]*partvz_z3[ip].x/ntp+tao_p1[ip]*partvz_z4[ip].x/ntp+tao_p2[ip]*partvz_z5[ip].x/ntp));	

		phi_vx_xz[ip] = phi_vx_xz[ip] + dt*(-a_x[ix]*phi_vx_xz[ip] - b_x[ix]*(
										(eta_p1[ip]-2*eta_s1[ip])*partvx_x1[ip].x/ntp + (eta_p2[ip]-2*eta_s2[ip])*partvx_x2[ip].x/ntp + (eta_p3[ip]-2*eta_s3[ip])*partvx_x3[ip].x/ntp
														+(tao_p1[ip]-2*tao_s1[ip])*partvx_x4[ip].x/ntp + (tao_p2[ip]-2*tao_s2[ip])*partvx_x5[ip].x/ntp));
		phi_vy_yz[ip] = phi_vy_yz[ip] + dt*(-a_y[iy]*phi_vy_yz[ip] - b_y[iy]*(
										(eta_p1[ip]-2*eta_s1[ip])*partvy_y1[ip].x/ntp + (eta_p2[ip]-2*eta_s2[ip])*partvy_y2[ip].x/ntp + (eta_p3[ip]-2*eta_s3[ip])*partvy_y3[ip].x/ntp
														+(tao_p1[ip]-2*tao_s1[ip])*partvy_y4[ip].x/ntp + (tao_p2[ip]-2*tao_s2[ip])*partvy_y5[ip].x/ntp));
       
		phi_vy_yy[ip] = phi_vy_yy[ip] + dt*(-a_y[iy]*phi_vy_yy[ip] - b_y[iy]*(
					eta_p1[ip]*partvy_y1[ip].x/ntp+eta_p2[ip]*partvy_y2[ip].x/ntp+eta_p3[ip]*partvy_y3[ip].x/ntp+tao_p1[ip]*partvy_y4[ip].x/ntp+tao_p2[ip]*partvy_y5[ip].x/ntp));	

		phi_vx_xy[ip] = phi_vx_xy[ip] + dt*(-a_x[ix]*phi_vx_xy[ip] - b_x[ix]*(
										(eta_p1[ip]-2*eta_s1[ip])*partvx_x1[ip].x/ntp + (eta_p2[ip]-2*eta_s2[ip])*partvx_x2[ip].x/ntp + (eta_p3[ip]-2*eta_s3[ip])*partvx_x3[ip].x/ntp
														+(tao_p1[ip]-2*tao_s1[ip])*partvx_x4[ip].x/ntp + (tao_p2[ip]-2*tao_s2[ip])*partvx_x5[ip].x/ntp));

		phi_vz_zy[ip] = phi_vz_zy[ip] + dt*(-a_z[iz]*phi_vz_zy[ip] - b_z[iz]*(
										(eta_p1[ip]-2*eta_s1[ip])*partvz_z1[ip].x/ntp + (eta_p2[ip]-2*eta_s2[ip])*partvz_z2[ip].x/ntp + (eta_p3[ip]-2*eta_s3[ip])*partvz_z3[ip].x/ntp
														+(tao_p1[ip]-2*tao_s1[ip])*partvz_z4[ip].x/ntp + (tao_p2[ip]-2*tao_s2[ip])*partvz_z5[ip].x/ntp));

		pxx[ip] = pxx[ip] + dt*(
								1.0/gammax[ix]*(eta_p1[ip]*partvx_x1[ip].x/ntp+eta_p2[ip]*partvx_x2[ip].x/ntp+eta_p3[ip]*partvx_x3[ip].x/ntp+tao_p1[ip]*partvx_x4[ip].x/ntp+tao_p2[ip]*partvx_x5[ip].x/ntp)+
								1.0/gammay[iy]*((eta_p1[ip]-2*eta_s1[ip])*partvy_y1[ip].x/ntp+(eta_p2[ip]-2*eta_s2[ip])*partvy_y2[ip].x/ntp+(eta_p3[ip]-2*eta_s3[ip])*partvy_y3[ip].x/ntp+(tao_p1[ip]-2*tao_s1[ip])*partvy_y4[ip].x/ntp+(tao_p2[ip]-2*tao_s2[ip])*partvy_y5[ip].x/ntp)+
								1.0/gammaz[iz]*((eta_p1[ip]-2*eta_s1[ip])*partvz_z1[ip].x/ntp+(eta_p2[ip]-2*eta_s2[ip])*partvz_z2[ip].x/ntp+(eta_p3[ip]-2*eta_s3[ip])*partvz_z3[ip].x/ntp+(tao_p1[ip]-2*tao_s1[ip])*partvz_z4[ip].x/ntp+(tao_p2[ip]-2*tao_s2[ip])*partvz_z5[ip].x/ntp)
							    +(phi_vx_xx[ip]+phi_vy_yx[ip]+phi_vz_zx[ip])
							);

		pyy[ip] = pyy[ip] + dt*(
								1.0/gammax[ix]*((eta_p1[ip]-2*eta_s1[ip])*partvx_x1[ip].x/ntp+(eta_p2[ip]-2*eta_s2[ip])*partvx_x2[ip].x/ntp+(eta_p3[ip]-2*eta_s3[ip])*partvx_x3[ip].x/ntp+(tao_p1[ip]-2*tao_s1[ip])*partvx_x4[ip].x/ntp+(tao_p2[ip]-2*tao_s2[ip])*partvx_x5[ip].x/ntp)+
								1.0/gammay[iy]*(eta_p1[ip]*partvy_y1[ip].x/ntp+eta_p2[ip]*partvy_y2[ip].x/ntp+eta_p3[ip]*partvy_y3[ip].x/ntp+tao_p1[ip]*partvy_y4[ip].x/ntp+tao_p2[ip]*partvy_y5[ip].x/ntp)+
								1.0/gammaz[iz]*((eta_p1[ip]-2*eta_s1[ip])*partvz_z1[ip].x/ntp+(eta_p2[ip]-2*eta_s2[ip])*partvz_z2[ip].x/ntp+(eta_p3[ip]-2*eta_s3[ip])*partvz_z3[ip].x/ntp+(tao_p1[ip]-2*tao_s1[ip])*partvz_z4[ip].x/ntp+(tao_p2[ip]-2*tao_s2[ip])*partvz_z5[ip].x/ntp)
							    +(phi_vx_xy[ip]+phi_vy_yy[ip]+phi_vz_zy[ip])
							);

		pzz[ip] = pzz[ip] + dt*(
								1.0/gammax[ix]*((eta_p1[ip]-2*eta_s1[ip])*partvx_x1[ip].x/ntp+(eta_p2[ip]-2*eta_s2[ip])*partvx_x2[ip].x/ntp+(eta_p3[ip]-2*eta_s3[ip])*partvx_x3[ip].x/ntp+(tao_p1[ip]-2*tao_s1[ip])*partvx_x4[ip].x/ntp+(tao_p2[ip]-2*tao_s2[ip])*partvx_x5[ip].x/ntp)+
								1.0/gammay[iy]*((eta_p1[ip]-2*eta_s1[ip])*partvy_y1[ip].x/ntp+(eta_p2[ip]-2*eta_s2[ip])*partvy_y2[ip].x/ntp+(eta_p3[ip]-2*eta_s3[ip])*partvy_y3[ip].x/ntp+(tao_p1[ip]-2*tao_s1[ip])*partvy_y4[ip].x/ntp+(tao_p2[ip]-2*tao_s2[ip])*partvy_y5[ip].x/ntp)+
								1.0/gammaz[iz]*(eta_p1[ip]*partvz_z1[ip].x/ntp+eta_p2[ip]*partvz_z2[ip].x/ntp+eta_p3[ip]*partvz_z3[ip].x/ntp+tao_p1[ip]*partvz_z4[ip].x/ntp+tao_p2[ip]*partvz_z5[ip].x/ntp)
							     +(phi_vx_xz[ip]+phi_vy_yz[ip]+phi_vz_zz[ip])
							);



		phi_vx_z[ip] = phi_vx_z[ip] + dt*(-0.5*(a_z[iz]+a_z[iz+1])*phi_vx_z[ip] - 0.5*(b_z[iz]+b_z[iz+1])*(
									0.125*(eta_s1[ip]+eta_s1[ip+1]+eta_s1[ip+ntz]+eta_s1[ip+1+ntz]+eta_s1[ip+nty*ntz]+eta_s1[ip+nty*ntz+1]+eta_s1[ip+nty*ntz+ntz]+eta_s1[ip+nty*ntz+1+ntz])*(partvx_z1[ip].x)/ntp+
									0.125*(eta_s2[ip]+eta_s2[ip+1]+eta_s2[ip+ntz]+eta_s2[ip+1+ntz]+eta_s2[ip+nty*ntz]+eta_s2[ip+nty*ntz+1]+eta_s2[ip+nty*ntz+ntz]+eta_s2[ip+nty*ntz+1+ntz])*(partvx_z2[ip].x)/ntp+
									0.125*(eta_s3[ip]+eta_s3[ip+1]+eta_s3[ip+ntz]+eta_s3[ip+1+ntz]+eta_s3[ip+nty*ntz]+eta_s3[ip+nty*ntz+1]+eta_s3[ip+nty*ntz+ntz]+eta_s3[ip+nty*ntz+1+ntz])*(partvx_z3[ip].x)/ntp+
									0.125*(tao_s1[ip]+tao_s1[ip+1]+tao_s1[ip+ntz]+tao_s1[ip+1+ntz]+tao_s1[ip+nty*ntz]+tao_s1[ip+nty*ntz+1]+tao_s1[ip+nty*ntz+ntz]+tao_s1[ip+nty*ntz+1+ntz])*(partvx_z4[ip].x)/ntp+
									0.125*(tao_s2[ip]+tao_s2[ip+1]+tao_s2[ip+ntz]+tao_s2[ip+1+ntz]+tao_s2[ip+nty*ntz]+tao_s2[ip+nty*ntz+1]+tao_s2[ip+nty*ntz+ntz]+tao_s2[ip+nty*ntz+1+ntz])*(partvx_z5[ip].x)/ntp 
																											)
										);	

		phi_vz_x[ip] = phi_vz_x[ip] + dt*(-0.5*(a_x[ix]+a_x[ix+1])*phi_vz_x[ip] - 0.5*(b_x[ix]+b_x[ix+1])*(
									0.125*(eta_s1[ip]+eta_s1[ip+1]+eta_s1[ip+ntz]+eta_s1[ip+1+ntz]+eta_s1[ip+nty*ntz]+eta_s1[ip+nty*ntz+1]+eta_s1[ip+nty*ntz+ntz]+eta_s1[ip+nty*ntz+1+ntz])*(partvz_x1[ip].x)/ntp+
									0.125*(eta_s2[ip]+eta_s2[ip+1]+eta_s2[ip+ntz]+eta_s2[ip+1+ntz]+eta_s2[ip+nty*ntz]+eta_s2[ip+nty*ntz+1]+eta_s2[ip+nty*ntz+ntz]+eta_s2[ip+nty*ntz+1+ntz])*(partvz_x2[ip].x)/ntp+
									0.125*(eta_s3[ip]+eta_s3[ip+1]+eta_s3[ip+ntz]+eta_s3[ip+1+ntz]+eta_s3[ip+nty*ntz]+eta_s3[ip+nty*ntz+1]+eta_s3[ip+nty*ntz+ntz]+eta_s3[ip+nty*ntz+1+ntz])*(partvz_x3[ip].x)/ntp+
									0.125*(tao_s1[ip]+tao_s1[ip+1]+tao_s1[ip+ntz]+tao_s1[ip+1+ntz]+tao_s1[ip+nty*ntz]+tao_s1[ip+nty*ntz+1]+tao_s1[ip+nty*ntz+ntz]+tao_s1[ip+nty*ntz+1+ntz])*(partvz_x4[ip].x)/ntp+
									0.125*(tao_s2[ip]+tao_s2[ip+1]+tao_s2[ip+ntz]+tao_s2[ip+1+ntz]+tao_s2[ip+nty*ntz]+tao_s2[ip+nty*ntz+1]+tao_s2[ip+nty*ntz+ntz]+tao_s2[ip+nty*ntz+1+ntz])*(partvz_x5[ip].x)/ntp
																											)
										);
		phi_vx_y[ip] = phi_vx_y[ip] + dt*(-0.5*(a_y[iy]+a_y[iy+1])*phi_vx_y[ip] - 0.5*(b_y[iy]+b_y[iy+1])*(
									0.125*(eta_s1[ip]+eta_s1[ip+1]+eta_s1[ip+ntz]+eta_s1[ip+1+ntz]+eta_s1[ip+nty*ntz]+eta_s1[ip+nty*ntz+1]+eta_s1[ip+nty*ntz+ntz]+eta_s1[ip+nty*ntz+1+ntz])*(partvx_y1[ip].x)/ntp+
									0.125*(eta_s2[ip]+eta_s2[ip+1]+eta_s2[ip+ntz]+eta_s2[ip+1+ntz]+eta_s2[ip+nty*ntz]+eta_s2[ip+nty*ntz+1]+eta_s2[ip+nty*ntz+ntz]+eta_s2[ip+nty*ntz+1+ntz])*(partvx_y2[ip].x)/ntp+
									0.125*(eta_s3[ip]+eta_s3[ip+1]+eta_s3[ip+ntz]+eta_s3[ip+1+ntz]+eta_s3[ip+nty*ntz]+eta_s3[ip+nty*ntz+1]+eta_s3[ip+nty*ntz+ntz]+eta_s3[ip+nty*ntz+1+ntz])*(partvx_y3[ip].x)/ntp+
									0.125*(tao_s1[ip]+tao_s1[ip+1]+tao_s1[ip+ntz]+tao_s1[ip+1+ntz]+tao_s1[ip+nty*ntz]+tao_s1[ip+nty*ntz+1]+tao_s1[ip+nty*ntz+ntz]+tao_s1[ip+nty*ntz+1+ntz])*(partvx_y4[ip].x)/ntp+
									0.125*(tao_s2[ip]+tao_s2[ip+1]+tao_s2[ip+ntz]+tao_s2[ip+1+ntz]+tao_s2[ip+nty*ntz]+tao_s2[ip+nty*ntz+1]+tao_s2[ip+nty*ntz+ntz]+tao_s2[ip+nty*ntz+1+ntz])*(partvx_y5[ip].x)/ntp
																											)
										);


		phi_vy_x[ip] = phi_vy_x[ip] + dt*(-0.5*(a_x[ix]+a_x[ix+1])*phi_vy_x[ip] - 0.5*(b_x[ix]+b_x[ix+1])*(
									0.125*(eta_s1[ip]+eta_s1[ip+1]+eta_s1[ip+ntz]+eta_s1[ip+1+ntz]+eta_s1[ip+nty*ntz]+eta_s1[ip+nty*ntz+1]+eta_s1[ip+nty*ntz+ntz]+eta_s1[ip+nty*ntz+1+ntz])*(partvy_x1[ip].x)/ntp+
									0.125*(eta_s2[ip]+eta_s2[ip+1]+eta_s2[ip+ntz]+eta_s2[ip+1+ntz]+eta_s2[ip+nty*ntz]+eta_s2[ip+nty*ntz+1]+eta_s2[ip+nty*ntz+ntz]+eta_s2[ip+nty*ntz+1+ntz])*(partvy_x2[ip].x)/ntp+
									0.125*(eta_s3[ip]+eta_s3[ip+1]+eta_s3[ip+ntz]+eta_s3[ip+1+ntz]+eta_s3[ip+nty*ntz]+eta_s3[ip+nty*ntz+1]+eta_s3[ip+nty*ntz+ntz]+eta_s3[ip+nty*ntz+1+ntz])*(partvy_x3[ip].x)/ntp+
									0.125*(tao_s1[ip]+tao_s1[ip+1]+tao_s1[ip+ntz]+tao_s1[ip+1+ntz]+tao_s1[ip+nty*ntz]+tao_s1[ip+nty*ntz+1]+tao_s1[ip+nty*ntz+ntz]+tao_s1[ip+nty*ntz+1+ntz])*(partvy_x4[ip].x)/ntp+
									0.125*(tao_s2[ip]+tao_s2[ip+1]+tao_s2[ip+ntz]+tao_s2[ip+1+ntz]+tao_s2[ip+nty*ntz]+tao_s2[ip+nty*ntz+1]+tao_s2[ip+nty*ntz+ntz]+tao_s2[ip+nty*ntz+1+ntz])*(partvy_x5[ip].x)/ntp 
																											)
										);	

		phi_vy_z[ip] = phi_vy_z[ip] + dt*(-0.5*(a_z[iz]+a_z[iz+1])*phi_vy_z[ip] - 0.5*(b_z[iz]+b_z[iz+1])*(
									0.125*(eta_s1[ip]+eta_s1[ip+1]+eta_s1[ip+ntz]+eta_s1[ip+1+ntz]+eta_s1[ip+nty*ntz]+eta_s1[ip+nty*ntz+1]+eta_s1[ip+nty*ntz+ntz]+eta_s1[ip+nty*ntz+1+ntz])*(partvy_z1[ip].x)/ntp+
									0.125*(eta_s2[ip]+eta_s2[ip+1]+eta_s2[ip+ntz]+eta_s2[ip+1+ntz]+eta_s2[ip+nty*ntz]+eta_s2[ip+nty*ntz+1]+eta_s2[ip+nty*ntz+ntz]+eta_s2[ip+nty*ntz+1+ntz])*(partvy_z2[ip].x)/ntp+
									0.125*(eta_s3[ip]+eta_s3[ip+1]+eta_s3[ip+ntz]+eta_s3[ip+1+ntz]+eta_s3[ip+nty*ntz]+eta_s3[ip+nty*ntz+1]+eta_s3[ip+nty*ntz+ntz]+eta_s3[ip+nty*ntz+1+ntz])*(partvy_z3[ip].x)/ntp+
									0.125*(tao_s1[ip]+tao_s1[ip+1]+tao_s1[ip+ntz]+tao_s1[ip+1+ntz]+tao_s1[ip+nty*ntz]+tao_s1[ip+nty*ntz+1]+tao_s1[ip+nty*ntz+ntz]+tao_s1[ip+nty*ntz+1+ntz])*(partvy_z4[ip].x)/ntp+
									0.125*(tao_s2[ip]+tao_s2[ip+1]+tao_s2[ip+ntz]+tao_s2[ip+1+ntz]+tao_s2[ip+nty*ntz]+tao_s2[ip+nty*ntz+1]+tao_s2[ip+nty*ntz+ntz]+tao_s2[ip+nty*ntz+1+ntz])*(partvy_z5[ip].x)/ntp 
																											)
										);	

		phi_vz_y[ip] = phi_vz_y[ip] + dt*(-0.5*(a_y[iy]+a_y[iy+1])*phi_vz_y[ip] - 0.5*(b_y[iy]+b_y[iy+1])*(
									0.125*(eta_s1[ip]+eta_s1[ip+1]+eta_s1[ip+ntz]+eta_s1[ip+1+ntz]+eta_s1[ip+nty*ntz]+eta_s1[ip+nty*ntz+1]+eta_s1[ip+nty*ntz+ntz]+eta_s1[ip+nty*ntz+1+ntz])*(partvz_y1[ip].x)/ntp+
								    0.125*(eta_s2[ip]+eta_s2[ip+1]+eta_s2[ip+ntz]+eta_s2[ip+1+ntz]+eta_s2[ip+nty*ntz]+eta_s2[ip+nty*ntz+1]+eta_s2[ip+nty*ntz+ntz]+eta_s2[ip+nty*ntz+1+ntz])*(partvz_y2[ip].x)/ntp+
									0.125*(eta_s3[ip]+eta_s3[ip+1]+eta_s3[ip+ntz]+eta_s3[ip+1+ntz]+eta_s3[ip+nty*ntz]+eta_s3[ip+nty*ntz+1]+eta_s3[ip+nty*ntz+ntz]+eta_s3[ip+nty*ntz+1+ntz])*(partvz_y3[ip].x)/ntp+
									0.125*(tao_s1[ip]+tao_s1[ip+1]+tao_s1[ip+ntz]+tao_s1[ip+1+ntz]+tao_s1[ip+nty*ntz]+tao_s1[ip+nty*ntz+1]+tao_s1[ip+nty*ntz+ntz]+tao_s1[ip+nty*ntz+1+ntz])*(partvz_y4[ip].x)/ntp+
									0.125*(tao_s2[ip]+tao_s2[ip+1]+tao_s2[ip+ntz]+tao_s2[ip+1+ntz]+tao_s2[ip+nty*ntz]+tao_s2[ip+nty*ntz+1]+tao_s2[ip+nty*ntz+ntz]+tao_s2[ip+nty*ntz+1+ntz])*(partvz_y5[ip].x)/ntp
																											)
										);

		pxz[ip] = pxz[ip] + dt*(
								1.0/(0.5*(gammaz[iz]+gammaz[iz+1]))*(
									0.125*(eta_s1[ip]+eta_s1[ip+1]+eta_s1[ip+ntz]+eta_s1[ip+1+ntz]+eta_s1[ip+nty*ntz]+eta_s1[ip+nty*ntz+1]+eta_s1[ip+nty*ntz+ntz]+eta_s1[ip+nty*ntz+1+ntz])*(partvx_z1[ip].x)/ntp+
									0.125*(eta_s2[ip]+eta_s2[ip+1]+eta_s2[ip+ntz]+eta_s2[ip+1+ntz]+eta_s2[ip+nty*ntz]+eta_s2[ip+nty*ntz+1]+eta_s2[ip+nty*ntz+ntz]+eta_s2[ip+nty*ntz+1+ntz])*(partvx_z2[ip].x)/ntp+
									0.125*(eta_s3[ip]+eta_s3[ip+1]+eta_s3[ip+ntz]+eta_s3[ip+1+ntz]+eta_s3[ip+nty*ntz]+eta_s3[ip+nty*ntz+1]+eta_s3[ip+nty*ntz+ntz]+eta_s3[ip+nty*ntz+1+ntz])*(partvx_z3[ip].x)/ntp+
									0.125*(tao_s1[ip]+tao_s1[ip+1]+tao_s1[ip+ntz]+tao_s1[ip+1+ntz]+tao_s1[ip+nty*ntz]+tao_s1[ip+nty*ntz+1]+tao_s1[ip+nty*ntz+ntz]+tao_s1[ip+nty*ntz+1+ntz])*(partvx_z4[ip].x)/ntp+
									0.125*(tao_s2[ip]+tao_s2[ip+1]+tao_s2[ip+ntz]+tao_s2[ip+1+ntz]+tao_s2[ip+nty*ntz]+tao_s2[ip+nty*ntz+1]+tao_s2[ip+nty*ntz+ntz]+tao_s2[ip+nty*ntz+1+ntz])*(partvx_z5[ip].x)/ntp
																	) + phi_vx_z[ip] +
							   1.0/(0.5*(gammax[ix]+gammax[ix+1]))*(
									0.125*(eta_s1[ip]+eta_s1[ip+1]+eta_s1[ip+ntz]+eta_s1[ip+1+ntz]+eta_s1[ip+nty*ntz]+eta_s1[ip+nty*ntz+1]+eta_s1[ip+nty*ntz+ntz]+eta_s1[ip+nty*ntz+1+ntz])*(partvz_x1[ip].x)/ntp+
									0.125*(eta_s2[ip]+eta_s2[ip+1]+eta_s2[ip+ntz]+eta_s2[ip+1+ntz]+eta_s2[ip+nty*ntz]+eta_s2[ip+nty*ntz+1]+eta_s2[ip+nty*ntz+ntz]+eta_s2[ip+nty*ntz+1+ntz])*(partvz_x2[ip].x)/ntp+
									0.125*(eta_s3[ip]+eta_s3[ip+1]+eta_s3[ip+ntz]+eta_s3[ip+1+ntz]+eta_s3[ip+nty*ntz]+eta_s3[ip+nty*ntz+1]+eta_s3[ip+nty*ntz+ntz]+eta_s3[ip+nty*ntz+1+ntz])*(partvz_x3[ip].x)/ntp+
									0.125*(tao_s1[ip]+tao_s1[ip+1]+tao_s1[ip+ntz]+tao_s1[ip+1+ntz]+tao_s1[ip+nty*ntz]+tao_s1[ip+nty*ntz+1]+tao_s1[ip+nty*ntz+ntz]+tao_s1[ip+nty*ntz+1+ntz])*(partvz_x4[ip].x)/ntp+
									0.125*(tao_s2[ip]+tao_s2[ip+1]+tao_s2[ip+ntz]+tao_s2[ip+1+ntz]+tao_s2[ip+nty*ntz]+tao_s2[ip+nty*ntz+1]+tao_s2[ip+nty*ntz+ntz]+tao_s2[ip+nty*ntz+1+ntz])*(partvz_x5[ip].x)/ntp
																    ) + phi_vz_x[ip]
								);

		pyz[ip] = pyz[ip] + dt*(
								1.0/(0.5*(gammaz[iz]+gammaz[iz+1]))*(
									0.125*(eta_s1[ip]+eta_s1[ip+1]+eta_s1[ip+ntz]+eta_s1[ip+1+ntz]+eta_s1[ip+nty*ntz]+eta_s1[ip+nty*ntz+1]+eta_s1[ip+nty*ntz+ntz]+eta_s1[ip+nty*ntz+1+ntz])*(partvy_z1[ip].x)/ntp+
									0.125*(eta_s2[ip]+eta_s2[ip+1]+eta_s2[ip+ntz]+eta_s2[ip+1+ntz]+eta_s2[ip+nty*ntz]+eta_s2[ip+nty*ntz+1]+eta_s2[ip+nty*ntz+ntz]+eta_s2[ip+nty*ntz+1+ntz])*(partvy_z2[ip].x)/ntp+
									0.125*(eta_s3[ip]+eta_s3[ip+1]+eta_s3[ip+ntz]+eta_s3[ip+1+ntz]+eta_s3[ip+nty*ntz]+eta_s3[ip+nty*ntz+1]+eta_s3[ip+nty*ntz+ntz]+eta_s3[ip+nty*ntz+1+ntz])*(partvy_z3[ip].x)/ntp+
									0.125*(tao_s1[ip]+tao_s1[ip+1]+tao_s1[ip+ntz]+tao_s1[ip+1+ntz]+tao_s1[ip+nty*ntz]+tao_s1[ip+nty*ntz+1]+tao_s1[ip+nty*ntz+ntz]+tao_s1[ip+nty*ntz+1+ntz])*(partvy_z4[ip].x)/ntp+
									0.125*(tao_s2[ip]+tao_s2[ip+1]+tao_s2[ip+ntz]+tao_s2[ip+1+ntz]+tao_s2[ip+nty*ntz]+tao_s2[ip+nty*ntz+1]+tao_s2[ip+nty*ntz+ntz]+tao_s2[ip+nty*ntz+1+ntz])*(partvy_z5[ip].x)/ntp
																	) + phi_vy_z[ip] +
								1.0/(0.5*(gammay[iy]+gammay[iy+1]))*(
									0.125*(eta_s1[ip]+eta_s1[ip+1]+eta_s1[ip+ntz]+eta_s1[ip+1+ntz]+eta_s1[ip+nty*ntz]+eta_s1[ip+nty*ntz+1]+eta_s1[ip+nty*ntz+ntz]+eta_s1[ip+nty*ntz+1+ntz])*(partvz_y1[ip].x)/ntp+
								    0.125*(eta_s2[ip]+eta_s2[ip+1]+eta_s2[ip+ntz]+eta_s2[ip+1+ntz]+eta_s2[ip+nty*ntz]+eta_s2[ip+nty*ntz+1]+eta_s2[ip+nty*ntz+ntz]+eta_s2[ip+nty*ntz+1+ntz])*(partvz_y2[ip].x)/ntp+
									0.125*(eta_s3[ip]+eta_s3[ip+1]+eta_s3[ip+ntz]+eta_s3[ip+1+ntz]+eta_s3[ip+nty*ntz]+eta_s3[ip+nty*ntz+1]+eta_s3[ip+nty*ntz+ntz]+eta_s3[ip+nty*ntz+1+ntz])*(partvz_y3[ip].x)/ntp+
									0.125*(tao_s1[ip]+tao_s1[ip+1]+tao_s1[ip+ntz]+tao_s1[ip+1+ntz]+tao_s1[ip+nty*ntz]+tao_s1[ip+nty*ntz+1]+tao_s1[ip+nty*ntz+ntz]+tao_s1[ip+nty*ntz+1+ntz])*(partvz_y4[ip].x)/ntp+
									0.125*(tao_s2[ip]+tao_s2[ip+1]+tao_s2[ip+ntz]+tao_s2[ip+1+ntz]+tao_s2[ip+nty*ntz]+tao_s2[ip+nty*ntz+1]+tao_s2[ip+nty*ntz+ntz]+tao_s2[ip+nty*ntz+1+ntz])*(partvz_y5[ip].x)/ntp
																	) + phi_vz_y[ip]
								);

		pxy[ip] = pxy[ip] + dt*(
								1.0/(0.5*(gammay[iy]+gammay[iy+1]))*(
									0.125*(eta_s1[ip]+eta_s1[ip+1]+eta_s1[ip+ntz]+eta_s1[ip+1+ntz]+eta_s1[ip+nty*ntz]+eta_s1[ip+nty*ntz+1]+eta_s1[ip+nty*ntz+ntz]+eta_s1[ip+nty*ntz+1+ntz])*(partvx_y1[ip].x)/ntp+
									0.125*(eta_s2[ip]+eta_s2[ip+1]+eta_s2[ip+ntz]+eta_s2[ip+1+ntz]+eta_s2[ip+nty*ntz]+eta_s2[ip+nty*ntz+1]+eta_s2[ip+nty*ntz+ntz]+eta_s2[ip+nty*ntz+1+ntz])*(partvx_y2[ip].x)/ntp+
									0.125*(eta_s3[ip]+eta_s3[ip+1]+eta_s3[ip+ntz]+eta_s3[ip+1+ntz]+eta_s3[ip+nty*ntz]+eta_s3[ip+nty*ntz+1]+eta_s3[ip+nty*ntz+ntz]+eta_s3[ip+nty*ntz+1+ntz])*(partvx_y3[ip].x)/ntp+
									0.125*(tao_s1[ip]+tao_s1[ip+1]+tao_s1[ip+ntz]+tao_s1[ip+1+ntz]+tao_s1[ip+nty*ntz]+tao_s1[ip+nty*ntz+1]+tao_s1[ip+nty*ntz+ntz]+tao_s1[ip+nty*ntz+1+ntz])*(partvx_y4[ip].x)/ntp+
									0.125*(tao_s2[ip]+tao_s2[ip+1]+tao_s2[ip+ntz]+tao_s2[ip+1+ntz]+tao_s2[ip+nty*ntz]+tao_s2[ip+nty*ntz+1]+tao_s2[ip+nty*ntz+ntz]+tao_s2[ip+nty*ntz+1+ntz])*(partvx_y5[ip].x)/ntp
																	) + phi_vx_y[ip] +
								1.0/(0.5*(gammax[ix]+gammax[ix+1]))*(
									0.125*(eta_s1[ip]+eta_s1[ip+1]+eta_s1[ip+ntz]+eta_s1[ip+1+ntz]+eta_s1[ip+nty*ntz]+eta_s1[ip+nty*ntz+1]+eta_s1[ip+nty*ntz+ntz]+eta_s1[ip+nty*ntz+1+ntz])*(partvy_x1[ip].x)/ntp+
									0.125*(eta_s2[ip]+eta_s2[ip+1]+eta_s2[ip+ntz]+eta_s2[ip+1+ntz]+eta_s2[ip+nty*ntz]+eta_s2[ip+nty*ntz+1]+eta_s2[ip+nty*ntz+ntz]+eta_s2[ip+nty*ntz+1+ntz])*(partvy_x2[ip].x)/ntp+
									0.125*(eta_s3[ip]+eta_s3[ip+1]+eta_s3[ip+ntz]+eta_s3[ip+1+ntz]+eta_s3[ip+nty*ntz]+eta_s3[ip+nty*ntz+1]+eta_s3[ip+nty*ntz+ntz]+eta_s3[ip+nty*ntz+1+ntz])*(partvy_x3[ip].x)/ntp+
									0.125*(tao_s1[ip]+tao_s1[ip+1]+tao_s1[ip+ntz]+tao_s1[ip+1+ntz]+tao_s1[ip+nty*ntz]+tao_s1[ip+nty*ntz+1]+tao_s1[ip+nty*ntz+ntz]+tao_s1[ip+nty*ntz+1+ntz])*(partvy_x4[ip].x)/ntp+
									0.125*(tao_s2[ip]+tao_s2[ip+1]+tao_s2[ip+ntz]+tao_s2[ip+1+ntz]+tao_s2[ip+nty*ntz]+tao_s2[ip+nty*ntz+1]+tao_s2[ip+nty*ntz+ntz]+tao_s2[ip+nty*ntz+1+ntz])*(partvy_x5[ip].x)/ntp
																	) + phi_vy_x[ip]
								);





	}
	
	
	__syncthreads();	
}


__global__ void cuda_kernel_calculate_v
(
	int ntx, int nty, int ntz, int ntp, float dt, 
	float *rho, float *vx, float *vz, float *vy,
	float *gammax, float *a_x, float *b_x,
	float *gammay, float *a_y, float *b_y,
	float *gammaz, float *a_z, float *b_z,
	float *phi_pxx_x, float *phi_pxy_y, float *phi_pxz_z,
	float *phi_pxy_x, float *phi_pyy_y, float *phi_pyz_z, 
	float *phi_pxz_x, float *phi_pyz_y, float *phi_pzz_z,
	cufftComplex *partx1, cufftComplex *partz1, cufftComplex *party1,
	cufftComplex *partx2, cufftComplex *partz2, cufftComplex *party2,
	cufftComplex *partx3, cufftComplex *partz3, cufftComplex *party3
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;

	int ip=ix*nty*ntz+iy*ntz+iz;

	
	if(iz>=0 && iz<ntz-1 && ix>=0 && ix<ntx-1&& iy>=0 && iy<nty-1)
	{

		phi_pxx_x[ip] = phi_pxx_x[ip] + 
					dt*(-0.5*(a_x[ix]+a_x[ix+1])*phi_pxx_x[ip]-0.5*(b_x[ix]+b_x[ix+1])*partx1[ip].x/ntp);

		phi_pxy_y[ip] = phi_pxy_y[ip] + 
							dt*(-a_y[iy]*phi_pxy_y[ip]-b_y[iy]*party1[ip].x/ntp);

		phi_pxz_z[ip] = phi_pxz_z[ip] + 
							dt*(-a_z[iz]*phi_pxz_z[ip]-b_z[iz]*partz1[ip].x/ntp);

		phi_pxy_x[ip] = phi_pxy_x[ip] + 
							dt*(-a_x[ix]*phi_pxy_x[ip]-b_x[ix]*partx3[ip].x/ntp);

		phi_pyy_y[ip] = phi_pyy_y[ip] + 
							dt*(-0.5*(a_y[iy]+a_y[iy+1])*phi_pyy_y[ip]-0.5*(b_y[iy]+b_y[iy+1])*party3[ip].x/ntp);

		phi_pyz_z[ip] = phi_pyz_z[ip] + 
							dt*(-a_z[iz]*phi_pyz_z[ip]-b_z[iz]*partz3[ip].x/ntp);

		phi_pxz_x[ip] = phi_pxz_x[ip] + 
							dt*(-a_x[ix]*phi_pxz_x[ip]-b_x[ix]*partx2[ip].x/ntp);

		phi_pyz_y[ip] = phi_pyz_y[ip] + 
							dt*(-a_y[iy]*phi_pyz_y[ip]-b_y[iy]*party2[ip].x/ntp);

		phi_pzz_z[ip] = phi_pzz_z[ip] + 
							dt*(-0.5*(a_z[iz]+a_z[iz+1])*phi_pzz_z[ip]-0.5*(b_z[iz]+b_z[iz+1])*partz2[ip].x/ntp);


		vx[ip] = vx[ip] + dt/(0.5*(rho[ip]+rho[ip+nty*ntz]))*
				(
					1.0/(0.5*(gammax[ix]+gammax[ix+1]))*partx1[ip].x/ntp + phi_pxx_x[ip] + 
					1.0/gammay[iy]*party1[ip].x/ntp + phi_pxy_y[ip] + 
					1.0/gammaz[iz]*partz1[ip].x/ntp + phi_pxz_z[ip] 
				);

		vy[ip] = vy[ip] + dt/(0.5*(rho[ip]+rho[ip+ntz]))*
				(
					1.0/gammax[ix]*partx3[ip].x/ntp + phi_pxy_x[ip] + 
					1.0/(0.5*(gammay[iy]+gammay[iy+1]))*party3[ip].x/ntp + phi_pyy_y[ip] + 
					1.0/gammaz[iz]*partz3[ip].x/ntp + phi_pyz_z[ip] 
				);


		vz[ip] = vz[ip] + dt/(0.5*(rho[ip]+rho[ip+1]))*
				(
					1.0/gammax[ix]*partx2[ip].x/ntp + phi_pxz_x[ip] + 
					1.0/gammay[iy]*party2[ip].x/ntp + phi_pyz_y[ip] + 
					1.0/(0.5*(gammaz[iz]+gammaz[iz+1]))*partz2[ip].x/ntp + phi_pzz_z[ip]
				);
			
	}
	__syncthreads();
}



__global__ void cuda_kernel_get_dv_renewed
(
	int ntx, int nty, int ntz, cufftComplex *outx, cufftComplex *outy, cufftComplex *outz, cufftComplex *dvx, cufftComplex *dvy, cufftComplex *dvz
)
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=by*BLOCK_HEIGHT+ty;
	int colt=bx*BLOCK_WIDTH+tx;
	int iz=colt/ntx;
	int ix=colt-iz*ntx;

	int ip=ix*nty*ntz+iy*ntz+iz;
	
	if(iz>=0 && iz<ntz && ix>=0 && ix<ntx && iy>=0 && iy<nty)
	{
		dvx[ip].x=outx[ip].x;
		dvx[ip].y=outx[ip].y;
		
		dvz[ip].x=outz[ip].x;
		dvz[ip].y=outz[ip].y;

		dvy[ip].x=outy[ip].x;
		dvy[ip].y=outy[ip].y;
	}
	__syncthreads();
}


extern "C"
void cuda_forward_acoustic_3D
( 
	int myid, int is, 
	int nt, int ntx, int nty, int ntz, int ntp, int nx, int ny, int nz, int pml, 
	float dx, float dy, float dz, float dt, float f0, float w0, float velp_max,
	float *rik, float *velp, float *gama_p,float *vels, float *gama_s, float *rho,
	struct Source ss[], struct MultiGPU plan[], int GPU_N, int rnmax, int rnx_max, int rny_max, int dr
)
{
	int i, it, ix, iy, iz;
	size_t size_model=sizeof(float)*ntp;
	char filename[150];
	FILE *fp;
	// define multistream  variable
	Multistream plans[GPU_N];
	
	float *tmp;
	tmp=(float*)malloc(sizeof(float)*ntp);


	// block size 16*16; 
	// grid size ntx/16*ntz/16
	dim3 dimBlock(BLOCK_WIDTH,BLOCK_HEIGHT);
	dim3 dimGrid((ntx*ntz+dimBlock.x-1)/dimBlock.x,(nty+dimBlock.y-1)/dimBlock.y);
	
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		// define streaming cufft handle (very important!!!)
		cudaStreamCreate(&plans[i].stream);	
		cufftSetStream(plan[i].PLAN_FORWARD, plans[i].stream);
		cufftSetStream(plan[i].PLAN_BACKWARD, plans[i].stream);
	}

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);			

		// copy the vectors from the host to the device
		
		cudaMemcpyAsync(plan[i].d_r_ix,ss[is+i].r_ix,sizeof(float)*rnmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_r_iy,ss[is+i].r_iy,sizeof(float)*rnmax,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_velp,velp,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_gama_p,gama_p,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_vels,vels,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_gama_s,gama_s,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_rho,rho,size_model,cudaMemcpyHostToDevice,plans[i].stream);
		cudaMemcpyAsync(plan[i].d_rik,rik,sizeof(float)*nt,cudaMemcpyHostToDevice,plans[i].stream);
	}
	
	
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		
		//===============define wavenumber variables============//		
		cuda_kernel_wavenumber<<<dimGrid,dimBlock,0,plans[i].stream>>>
		(
			ntx, nty, ntz, dx, dy, dz, plan[i].d_kx, plan[i].d_ky, plan[i].d_kz, plan[i].d_k,
			plan[i].d_kvx_x, plan[i].d_kvx_z, plan[i].d_kvx_y, plan[i].d_kvz_x, plan[i].d_kvz_z, plan[i].d_kvz_y,
			plan[i].d_kvy_x, plan[i].d_kvy_z, plan[i].d_kvy_y
		);

		//===============define viscoacoustic variables============//				
		cuda_kernel_viscoacoustic_parameters<<<dimGrid,dimBlock,0,plans[i].stream>>>
		(
			ntx, nty, ntz, dx, dy, dz, dt, w0,
			plan[i].d_velp, plan[i].d_vels, plan[i].d_rho, plan[i].d_k, plan[i].d_gama_p, plan[i].d_gama_s,
			plan[i].d_Ap1, plan[i].d_Ap2, plan[i].d_Ap3, 
			plan[i].d_tao_p1, plan[i].d_tao_p2, plan[i].d_eta_p1, plan[i].d_eta_p2, plan[i].d_eta_p3,
			plan[i].d_tao_s1, plan[i].d_tao_s2, plan[i].d_eta_s1, plan[i].d_eta_s2, plan[i].d_eta_s3
		);

		//===============PML parameters============//
		cuda_kernel_pml_parameters<<<dimGrid,dimBlock,0,plans[i].stream>>>
		(
			ntx, nty, ntz, pml, dx, dy, dz, dt, f0, velp_max,
			plan[i].d_gammax, plan[i].d_alphax, plan[i].d_Omegax, plan[i].d_a_x, plan[i].d_b_x,
			plan[i].d_gammay, plan[i].d_alphay, plan[i].d_Omegay, plan[i].d_a_y, plan[i].d_b_y,
			plan[i].d_gammaz, plan[i].d_alphaz, plan[i].d_Omegaz, plan[i].d_a_z, plan[i].d_b_z	
		);
		
		//===============initialization============//
		cuda_kernel_initialization<<<dimGrid,dimBlock,0,plans[i].stream>>>
		(
			ntx, nty, ntz, 
			plan[i].d_vx, plan[i].d_vy, plan[i].d_vz,
			plan[i].d_pxx, plan[i].d_pyy, plan[i].d_pzz, 
			plan[i].d_pxy, plan[i].d_pyz, plan[i].d_pxz,
			plan[i].d_phi_vx_xx, plan[i].d_phi_vz_zx, plan[i].d_phi_vy_yx, 
			plan[i].d_phi_vx_xy, plan[i].d_phi_vz_zy, plan[i].d_phi_vy_yy, 
			plan[i].d_phi_vx_xz, plan[i].d_phi_vz_zz, plan[i].d_phi_vy_yz, 
			plan[i].d_phi_vx_z, plan[i].d_phi_vz_x,plan[i].d_phi_vx_y, 
			plan[i].d_phi_vy_x,plan[i].d_phi_vy_z, plan[i].d_phi_vz_y,
			plan[i].d_phi_pxx_x, plan[i].d_phi_pxy_y, plan[i].d_phi_pxz_z,
			plan[i].d_phi_pxy_x, plan[i].d_phi_pyy_y, plan[i].d_phi_pyz_z, 
			plan[i].d_phi_pxz_x, plan[i].d_phi_pyz_y, plan[i].d_phi_pzz_z,
            plan[i].d_dvx, plan[i].d_dvy, plan[i].d_dvz,
            plan[i].d_partx1, plan[i].d_partz1, plan[i].d_party1,
            plan[i].d_partx2, plan[i].d_partz2, plan[i].d_party2,
            plan[i].d_partx3, plan[i].d_partz3, plan[i].d_party3,
            plan[i].d_partvx_x1, plan[i].d_partvx_x2, plan[i].d_partvx_x3, plan[i].d_partvx_x4, plan[i].d_partvx_x5,
			plan[i].d_partvz_z1, plan[i].d_partvz_z2, plan[i].d_partvz_z3, plan[i].d_partvz_z4, plan[i].d_partvz_z5,
			plan[i].d_partvy_y1, plan[i].d_partvy_y2, plan[i].d_partvy_y3, plan[i].d_partvy_y4, plan[i].d_partvy_y5,
            plan[i].d_partvx_y1, plan[i].d_partvx_y2, plan[i].d_partvx_y3, plan[i].d_partvx_y4, plan[i].d_partvx_y5,
            plan[i].d_partvy_x1, plan[i].d_partvy_x2, plan[i].d_partvy_x3, plan[i].d_partvy_x4, plan[i].d_partvy_x5,
			plan[i].d_partvy_z1, plan[i].d_partvy_z2, plan[i].d_partvy_z3, plan[i].d_partvy_z4, plan[i].d_partvy_z5,
			plan[i].d_partvz_y1, plan[i].d_partvz_y2, plan[i].d_partvz_y3, plan[i].d_partvz_y4, plan[i].d_partvz_y5,
            plan[i].d_partvx_z1, plan[i].d_partvx_z2, plan[i].d_partvx_z3, plan[i].d_partvx_z4, plan[i].d_partvx_z5,
			plan[i].d_partvz_x1, plan[i].d_partvz_x2, plan[i].d_partvz_x3, plan[i].d_partvz_x4, plan[i].d_partvz_x5
		);

	}	
	
	//===================time begin===========================//
	//===================time begin===========================//
	for(it=0;it<nt;it++)
	{
		for(i=0;i<GPU_N;i++)
		{
			cudaSetDevice(i);	
			
			
			//===============calculate k-space spatial derivative============//
			//===============calculate k-space spatial derivative============//
			
			cuda_kernel_p_real_to_complex<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, 
				plan[i].d_pxx, plan[i].d_pyy, plan[i].d_pzz, plan[i].d_pxy, plan[i].d_pxz, plan[i].d_pyz,
				plan[i].d_in_pxx, plan[i].d_in_pyy, plan[i].d_in_pzz, plan[i].d_in_pxy, plan[i].d_in_pxz, plan[i].d_in_pyz
			);
			
			cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_in_pxx,plan[i].d_outpxx,CUFFT_FORWARD);
			cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_in_pyy,plan[i].d_outpyy,CUFFT_FORWARD);
			cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_in_pzz,plan[i].d_outpzz,CUFFT_FORWARD);
			cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_in_pxy,plan[i].d_outpxy,CUFFT_FORWARD);
			cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_in_pxz,plan[i].d_outpxz,CUFFT_FORWARD);
			cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_in_pyz,plan[i].d_outpyz,CUFFT_FORWARD);
			
			cuda_kernel_operate_k_v<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, plan[i].d_outpxx, plan[i].d_outpxy, plan[i].d_outpxz, plan[i].d_inx, plan[i].d_iny, plan[i].d_inz, plan[i].d_kvz_x, plan[i].d_kvy_y, plan[i].d_kvz_z
			);	

			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partx1, CUFFT_INVERSE);		//dpxxdx
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_party1, CUFFT_INVERSE);		//dpxydy
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partz1, CUFFT_INVERSE);		//dpxzdz


			cuda_kernel_operate_k_v<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, plan[i].d_outpxz, plan[i].d_outpyz, plan[i].d_outpzz, plan[i].d_inx, plan[i].d_iny, plan[i].d_inz, plan[i].d_kvx_x, plan[i].d_kvy_y, plan[i].d_kvx_z
			);	

			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partx2, CUFFT_INVERSE);		//dpxzdx
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_party2, CUFFT_INVERSE);		//dpyzdy
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partz2, CUFFT_INVERSE);		//dpzzdz


			cuda_kernel_operate_k_v<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, plan[i].d_outpxy, plan[i].d_outpyy, plan[i].d_outpyz, plan[i].d_inx, plan[i].d_iny, plan[i].d_inz, plan[i].d_kvx_x, plan[i].d_kvz_y, plan[i].d_kvz_z
			);	

			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partx3, CUFFT_INVERSE);		//dpxydx
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_party3, CUFFT_INVERSE);		//dpyydy
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partz3, CUFFT_INVERSE);		//dpyzdz


			//===================calculate vx vy and vz==================//	
			//===================calculate vx vy and vz==================//	
				
			cuda_kernel_calculate_v<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
	           ntx,  nty,  ntz,  ntp,  dt, 
			   plan[i].d_rho, plan[i].d_vx, plan[i].d_vz, plan[i].d_vy,
	           plan[i].d_gammax, plan[i].d_a_x, plan[i].d_b_x,
	           plan[i].d_gammay, plan[i].d_a_y, plan[i].d_b_y,
	           plan[i].d_gammaz, plan[i].d_a_z, plan[i].d_b_z,
	           plan[i].d_phi_pxx_x, plan[i].d_phi_pxy_y, plan[i].d_phi_pxz_z,
	           plan[i].d_phi_pxy_x, plan[i].d_phi_pyy_y, plan[i].d_phi_pyz_z, 
	           plan[i].d_phi_pxz_x, plan[i].d_phi_pyz_y, plan[i].d_phi_pzz_z,
			   plan[i].d_partx1, plan[i].d_partz1, plan[i].d_party1,
			   plan[i].d_partx2, plan[i].d_partz2, plan[i].d_party2,
               plan[i].d_partx3, plan[i].d_partz3, plan[i].d_party3
			);


			//===============calculate k-space spatial derivatives============//		
			//===============calculate k-space spatial derivatives============//

			cuda_kernel_vxvyvz_real_to_complex<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, plan[i].d_vx, plan[i].d_vy, plan[i].d_vz, plan[i].d_inx, plan[i].d_iny, plan[i].d_inz
			);

			cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_inx,plan[i].d_outx,CUFFT_FORWARD);
			cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_iny,plan[i].d_outy,CUFFT_FORWARD);
			cufftExecC2C(plan[i].PLAN_FORWARD,plan[i].d_inz,plan[i].d_outz,CUFFT_FORWARD);

			
            ////////////////////////////////////	sigma xx yy zz dispersion_3 parts //////////////////////////
            cuda_kernel_operate_k_pxxpyypzz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outy, plan[i].d_outz, plan[i].d_dvx, plan[i].d_dvy, plan[i].d_dvz,
				plan[i].d_inx, plan[i].d_iny, plan[i].d_inz, plan[i].d_kvx_x, plan[i].d_kvy_y, plan[i].d_kvz_z, plan[i].d_Ap1, 0
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_x1, CUFFT_INVERSE);		//dvxdx, k^-0.5
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_y1, CUFFT_INVERSE);		//dvydy
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_z1, CUFFT_INVERSE);		//dvzdz


            cuda_kernel_operate_k_pxxpyypzz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outy, plan[i].d_outz, plan[i].d_dvx, plan[i].d_dvy, plan[i].d_dvz,
				plan[i].d_inx, plan[i].d_iny, plan[i].d_inz, plan[i].d_kvx_x, plan[i].d_kvy_y, plan[i].d_kvz_z, plan[i].d_Ap2, 0
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_x2, CUFFT_INVERSE);		//dvxdx, 1*
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_y2, CUFFT_INVERSE);		//dvydy
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_z2, CUFFT_INVERSE);		//dvzdz  


            cuda_kernel_operate_k_pxxpyypzz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outy, plan[i].d_outz, plan[i].d_dvx, plan[i].d_dvy, plan[i].d_dvz,
				plan[i].d_inx, plan[i].d_iny, plan[i].d_inz, plan[i].d_kvx_x, plan[i].d_kvy_y, plan[i].d_kvz_z, plan[i].d_Ap3, 0
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_x3, CUFFT_INVERSE);		//dvxdx, k^0.5*
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_y3, CUFFT_INVERSE);		//dvydy
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_z3, CUFFT_INVERSE);		//dvzdz

////////////////////////////////////	sigma xz zx dispersion_3 parts //////////////////////////

            cuda_kernel_operate_k_pxz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outz, plan[i].d_dvx, plan[i].d_dvz,
				plan[i].d_inx, plan[i].d_inz, plan[i].d_kvx_z, plan[i].d_kvz_x, plan[i].d_Ap1, 0
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_z1, CUFFT_INVERSE);		//dvxdz, k^-0.5*
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_x1, CUFFT_INVERSE);		//dvzdx  

            cuda_kernel_operate_k_pxz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outz, plan[i].d_dvx, plan[i].d_dvz,
				plan[i].d_inx, plan[i].d_inz, plan[i].d_kvx_z, plan[i].d_kvz_x, plan[i].d_Ap2, 0
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_z2, CUFFT_INVERSE);		//dvxdz, 1*
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_x2, CUFFT_INVERSE);		//dvzdx    

            cuda_kernel_operate_k_pxz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outz, plan[i].d_dvx, plan[i].d_dvz,
				plan[i].d_inx, plan[i].d_inz, plan[i].d_kvx_z, plan[i].d_kvz_x, plan[i].d_Ap3, 0
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_z3, CUFFT_INVERSE);		//dvxdz, k^0.5*
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_x3, CUFFT_INVERSE);		//dvzdx   

////////////////////////////////////	sigma xy yx dispersion_3 parts //////////////////////////

            cuda_kernel_operate_k_pxy<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outy, plan[i].d_dvx, plan[i].d_dvy,
				plan[i].d_inx, plan[i].d_iny, plan[i].d_kvx_y, plan[i].d_kvy_x, plan[i].d_Ap1, 0
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_y1, CUFFT_INVERSE);		//dvxdy, k^-0.5*
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_x1, CUFFT_INVERSE);		//dvydx        

            cuda_kernel_operate_k_pxy<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outy, plan[i].d_dvx, plan[i].d_dvy,
				plan[i].d_inx, plan[i].d_iny, plan[i].d_kvx_y, plan[i].d_kvy_x, plan[i].d_Ap2, 0
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_y2, CUFFT_INVERSE);		//dvxdy, 1*
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_x2, CUFFT_INVERSE);		//dvydx   

            cuda_kernel_operate_k_pxy<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outy, plan[i].d_dvx, plan[i].d_dvy,
				plan[i].d_inx, plan[i].d_iny, plan[i].d_kvx_y, plan[i].d_kvy_x, plan[i].d_Ap3, 0
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_y3, CUFFT_INVERSE);		//dvxdy, k^0.5
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_x3, CUFFT_INVERSE);		//dvydx        
     
////////////////////////////////////	sigma yz zy dispersion_3 parts //////////////////////////

            cuda_kernel_operate_k_pyz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outy, plan[i].d_outz, plan[i].d_dvy, plan[i].d_dvz,
				plan[i].d_iny, plan[i].d_inz, plan[i].d_kvy_z, plan[i].d_kvz_y, plan[i].d_Ap1, 0
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_z1, CUFFT_INVERSE);		//dvydz, k^-0.5
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_y1, CUFFT_INVERSE);		//dvzdy 

            cuda_kernel_operate_k_pyz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outy, plan[i].d_outz, plan[i].d_dvy, plan[i].d_dvz,
				plan[i].d_iny, plan[i].d_inz, plan[i].d_kvy_z, plan[i].d_kvz_y, plan[i].d_Ap2, 0
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_z2, CUFFT_INVERSE);		//dvydz, 1*
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_y2, CUFFT_INVERSE);		//dvzdy 

            cuda_kernel_operate_k_pyz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outy, plan[i].d_outz, plan[i].d_dvy, plan[i].d_dvz,
				plan[i].d_iny, plan[i].d_inz, plan[i].d_kvy_z, plan[i].d_kvz_y, plan[i].d_Ap3, 0
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_z3, CUFFT_INVERSE);		//dvydz, k^0.5
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_y3, CUFFT_INVERSE);		//dvzdy 

////////////////////////////////////	sigma xx yy zz amplitude-loss_2 parts //////////////////////////

            cuda_kernel_operate_k_pxxpyypzz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outy, plan[i].d_outz, plan[i].d_dvx, plan[i].d_dvy, plan[i].d_dvz,
				plan[i].d_inx, plan[i].d_iny, plan[i].d_inz, plan[i].d_kvx_x, plan[i].d_kvy_y, plan[i].d_kvz_z, plan[i].d_Ap1, 1
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_x4, CUFFT_INVERSE);		//dvxdx, k^-0.5
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_y4, CUFFT_INVERSE);		//dvydy
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_z4, CUFFT_INVERSE);		//dvzdz


            cuda_kernel_operate_k_pxxpyypzz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outy, plan[i].d_outz, plan[i].d_dvx, plan[i].d_dvy, plan[i].d_dvz,
				plan[i].d_inx, plan[i].d_iny, plan[i].d_inz, plan[i].d_kvx_x, plan[i].d_kvy_y, plan[i].d_kvz_z, plan[i].d_Ap2, 1
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_x5, CUFFT_INVERSE);		//dvxdx, 1*
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_y5, CUFFT_INVERSE);		//dvydy
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_z5, CUFFT_INVERSE);		//dvzdz 

////////////////////////////////////	sigma xz zx amplitude-loss_2 parts //////////////////////////

            cuda_kernel_operate_k_pxz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outz, plan[i].d_dvx, plan[i].d_dvz,
				plan[i].d_inx, plan[i].d_inz, plan[i].d_kvx_z, plan[i].d_kvz_x, plan[i].d_Ap1, 1
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_z4, CUFFT_INVERSE);		//dvxdz, k^-0.5*
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_x4, CUFFT_INVERSE);		//dvzdx  

            cuda_kernel_operate_k_pxz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outz, plan[i].d_dvx, plan[i].d_dvz,
				plan[i].d_inx, plan[i].d_inz, plan[i].d_kvx_z, plan[i].d_kvz_x, plan[i].d_Ap2, 1
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_z5, CUFFT_INVERSE);		//dvxdz, 1*
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_x5, CUFFT_INVERSE);		//dvzdx   

////////////////////////////////////	sigma xy yx amplitude-loss_2 parts //////////////////////////

            cuda_kernel_operate_k_pxy<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outy, plan[i].d_dvx, plan[i].d_dvy,
				plan[i].d_inx, plan[i].d_iny, plan[i].d_kvx_y, plan[i].d_kvy_x, plan[i].d_Ap1, 1
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_y4, CUFFT_INVERSE);		//dvxdy, k^-0.5*
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_x4, CUFFT_INVERSE);		//dvydx        

            cuda_kernel_operate_k_pxy<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outx, plan[i].d_outy, plan[i].d_dvx, plan[i].d_dvy,
				plan[i].d_inx, plan[i].d_iny, plan[i].d_kvx_y, plan[i].d_kvy_x, plan[i].d_Ap2, 1
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inx,plan[i].d_partvx_y5, CUFFT_INVERSE);		//dvxdy, 1*
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_x5, CUFFT_INVERSE);		//dvydx  

////////////////////////////////////	sigma yz zy amplitude-loss_2 parts //////////////////////////

            cuda_kernel_operate_k_pyz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outy, plan[i].d_outz, plan[i].d_dvy, plan[i].d_dvz,
				plan[i].d_iny, plan[i].d_inz, plan[i].d_kvy_z, plan[i].d_kvz_y, plan[i].d_Ap1, 1
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_z4, CUFFT_INVERSE);		//dvydz, k^-0.5
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_y4, CUFFT_INVERSE);		//dvzdy 

            cuda_kernel_operate_k_pyz<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, dt, 
				plan[i].d_outy, plan[i].d_outz, plan[i].d_dvy, plan[i].d_dvz,
				plan[i].d_iny, plan[i].d_inz, plan[i].d_kvy_z, plan[i].d_kvz_y, plan[i].d_Ap2, 1
			);	//0 or 1 here stand for parameter AorB, where 1 stand for the first order time derivative of k-space variables.

            cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_iny,plan[i].d_partvy_z5, CUFFT_INVERSE);		//dvydz, 1*
			cufftExecC2C(plan[i].PLAN_BACKWARD, plan[i].d_inz,plan[i].d_partvz_y5, CUFFT_INVERSE);		//dvzdy 



			//===================calculate p ==================//	
			//===================calculate p ==================//	
				
			cuda_kernel_calculate_p<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
			   ntx,  nty,  ntz,  ntp,  dt,
			   plan[i].d_pxx, plan[i].d_pyy, plan[i].d_pzz, plan[i].d_pxy, plan[i].d_pxz, plan[i].d_pyz,  
			   plan[i].d_tao_p1, plan[i].d_tao_s1,plan[i].d_tao_p2,plan[i].d_tao_s2,plan[i].d_eta_p1,plan[i].d_eta_s1,plan[i].d_eta_p2,plan[i].d_eta_s2,plan[i].d_eta_p3,plan[i].d_eta_s3,
			   plan[i].d_gammax, plan[i].d_a_x, plan[i].d_b_x,
			   plan[i].d_gammay, plan[i].d_a_y, plan[i].d_b_y,
			   plan[i].d_gammaz, plan[i].d_a_z, plan[i].d_b_z,
			   plan[i].d_phi_vx_xx, plan[i].d_phi_vz_zx, plan[i].d_phi_vy_yx, 
			   plan[i].d_phi_vx_xy, plan[i].d_phi_vz_zy, plan[i].d_phi_vy_yy, 
			   plan[i].d_phi_vx_xz, plan[i].d_phi_vz_zz, plan[i].d_phi_vy_yz, 
			   plan[i].d_phi_vx_z,  plan[i].d_phi_vz_x,  plan[i].d_phi_vx_y, 
			   plan[i].d_phi_vy_x,  plan[i].d_phi_vy_z,  plan[i].d_phi_vz_y,
			   plan[i].d_partvx_x1, plan[i].d_partvx_x2, plan[i].d_partvx_x3, plan[i].d_partvx_x4, plan[i].d_partvx_x5,
			   plan[i].d_partvz_z1, plan[i].d_partvz_z2, plan[i].d_partvz_z3, plan[i].d_partvz_z4, plan[i].d_partvz_z5,
			   plan[i].d_partvy_y1, plan[i].d_partvy_y2, plan[i].d_partvy_y3, plan[i].d_partvy_y4, plan[i].d_partvy_y5,
			   plan[i].d_partvx_z1, plan[i].d_partvx_z2, plan[i].d_partvx_z3, plan[i].d_partvx_z4, plan[i].d_partvx_z5,
			   plan[i].d_partvz_x1, plan[i].d_partvz_x2, plan[i].d_partvz_x3, plan[i].d_partvz_x4, plan[i].d_partvz_x5,
			   plan[i].d_partvx_y1, plan[i].d_partvx_y2, plan[i].d_partvx_y3, plan[i].d_partvx_y4, plan[i].d_partvx_y5,
			   plan[i].d_partvy_x1, plan[i].d_partvy_x2, plan[i].d_partvy_x3, plan[i].d_partvy_x4, plan[i].d_partvy_x5,
			   plan[i].d_partvy_z1, plan[i].d_partvy_z2, plan[i].d_partvy_z3, plan[i].d_partvy_z4, plan[i].d_partvy_z5,
	           plan[i].d_partvz_y1, plan[i].d_partvz_y2, plan[i].d_partvz_y3, plan[i].d_partvz_y4, plan[i].d_partvz_y5
			);

			cuda_kernel_forward_IO<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, ntp, pml, nt, it, dx, dy, dz, dt, 
				ss[is+i].s_ix, ss[is+i].s_iy, ss[is+i].s_iz, plan[i].d_rik,
				plan[i].d_record, plan[i].d_record2, plan[i].d_record3, plan[i].d_r_ix, plan[i].d_r_iy, ss[is+i].r_iz, rnmax, rnx_max, rny_max, dr, ss[is+i].r_n, 
				plan[i].d_pxx, plan[i].d_pyy, plan[i].d_pzz, plan[i].d_vx, plan[i].d_vy, plan[i].d_vz
			);


			cuda_kernel_get_dv_renewed<<<dimGrid,dimBlock,0,plans[i].stream>>>
			(
				ntx, nty, ntz, plan[i].d_outx, plan[i].d_outy, plan[i].d_outz, plan[i].d_dvx, plan[i].d_dvy, plan[i].d_dvz
			);


			//================updating wavefields==================//

							
			if(it%200==0 && myid==0 && i==0)
			{

					printf("forward using real model,is=%2d,it=%4d\n",is+i,it);		
/*			
				cudaMemcpyAsync(tmp,plan[i].d_p2,size_model,cudaMemcpyDeviceToHost,plans[i].stream);

				sprintf(filename,"./output/shot%dsnap%d.bin",is+i,it);
				fp=fopen(filename,"wb");
				for(ix=pml;ix<ntx-pml;ix++)
					for(iy=pml; iy<nty-pml; iy++)
						for(iz=pml; iz<ntz-pml; iz++)
						{
							fwrite(&tmp[ix*nty*ntz+iy*ntz+iz],sizeof(float),1,fp);
						}
				fclose(fp);			
*/
			}				
		}		//end of GPU_N loop
	}		//end of time loop	



	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);
		
		

			cudaMemcpyAsync(plan[i].record, plan[i].d_record,sizeof(float)*nt*rnmax,
												cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].record2, plan[i].d_record2,sizeof(float)*nt*rnmax,
												cudaMemcpyDeviceToHost,plans[i].stream);
			cudaMemcpyAsync(plan[i].record3, plan[i].d_record3,sizeof(float)*nt*rnmax,
												cudaMemcpyDeviceToHost,plans[i].stream);
						
		cudaDeviceSynchronize();
		cudaStreamDestroy(plans[i].stream);
	}
	
	free(tmp);
}


//=========================================================
//  Initializating the memory for variables in device
//  =======================================================
extern "C"
void cuda_Host_initialization
(
	int ntp, int nt, int rnmax, struct MultiGPU plan[], int GPU_N
)
{
	int i;
	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		memset(plan[i].pxx, 0, ntp*sizeof(float));
		memset(plan[i].pyy, 0, ntp*sizeof(float));
		memset(plan[i].pzz, 0, ntp*sizeof(float));
		memset(plan[i].pxy, 0, ntp*sizeof(float));
		memset(plan[i].pxz, 0, ntp*sizeof(float));
		memset(plan[i].pyz, 0, ntp*sizeof(float));
		memset(plan[i].vx, 0, ntp*sizeof(float));
		memset(plan[i].vy, 0, ntp*sizeof(float));
		memset(plan[i].vz, 0, ntp*sizeof(float));

		memset(plan[i].record, 0, nt*rnmax*sizeof(float));
		memset(plan[i].record2, 0, nt*rnmax*sizeof(float));
		memset(plan[i].record3, 0, nt*rnmax*sizeof(float));
	}
}


//=================================================//
//  Allocate the memory for variables in device
//  ================================================//
extern "C"
void cuda_Device_malloc
(
	int ntx, int nty, int ntz, int ntp, int nx, int ny, int nz, int nt, 
	int rnmax, struct MultiGPU plan[], int GPU_N
)
{
	int i;
	size_t size_model=sizeof(float)*ntp;

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cufftPlan3d(&plan[i].PLAN_FORWARD,ntx, nty, ntz,CUFFT_C2C);
		cufftPlan3d(&plan[i].PLAN_BACKWARD,ntx, nty, ntz,CUFFT_C2C);

		//===========Host======================//
		//===========Host======================//
		cudaMallocHost((void **)&plan[i].pxx, size_model);
		cudaMallocHost((void **)&plan[i].pyy, size_model);
		cudaMallocHost((void **)&plan[i].pzz, size_model);
		cudaMallocHost((void **)&plan[i].pxy, size_model);
		cudaMallocHost((void **)&plan[i].pxz, size_model);
		cudaMallocHost((void **)&plan[i].pyz, size_model);
		cudaMallocHost((void **)&plan[i].vx, size_model);
		cudaMallocHost((void **)&plan[i].vy, size_model);
		cudaMallocHost((void **)&plan[i].vz, size_model);
			
		cudaMallocHost((void **)&plan[i].record, sizeof(float)*rnmax*nt);
		cudaMallocHost((void **)&plan[i].record2, sizeof(float)*rnmax*nt);
		cudaMallocHost((void **)&plan[i].record3, sizeof(float)*rnmax*nt);
		
		//===========device======================//
		//===========device======================//
		cudaMalloc((void **)&plan[i].d_r_ix,sizeof(int)*rnmax);
		cudaMalloc((void **)&plan[i].d_r_iy,sizeof(int)*rnmax);
		cudaMalloc((void **)&plan[i].d_rik,sizeof(float)*nt);
		
		cudaMalloc((void **)&plan[i].d_velp, size_model);
		cudaMalloc((void **)&plan[i].d_gama_p, size_model);	
		cudaMalloc((void **)&plan[i].d_vels, size_model);
		cudaMalloc((void **)&plan[i].d_gama_s, size_model);	
		cudaMalloc((void **)&plan[i].d_rho, size_model);	
		
		cudaMalloc((void **)&plan[i].d_pxx, size_model);	
		cudaMalloc((void **)&plan[i].d_pyy, size_model);
		cudaMalloc((void **)&plan[i].d_pzz, size_model);
		cudaMalloc((void **)&plan[i].d_pxy, size_model);
		cudaMalloc((void **)&plan[i].d_pxz, size_model);
		cudaMalloc((void **)&plan[i].d_pyz, size_model);
		cudaMalloc((void **)&plan[i].d_vx, size_model);	
		cudaMalloc((void **)&plan[i].d_vy, size_model);	
		cudaMalloc((void **)&plan[i].d_vz, size_model);	

////////////////   pml //////////////
		cudaMalloc((void **)&plan[i].d_gammax,sizeof(float)*ntx);
		cudaMalloc((void **)&plan[i].d_alphax,sizeof(float)*ntx);
		cudaMalloc((void **)&plan[i].d_Omegax,sizeof(float)*ntx);
		cudaMalloc((void **)&plan[i].d_a_x,sizeof(float)*ntx);
		cudaMalloc((void **)&plan[i].d_b_x,sizeof(float)*ntx);
		cudaMalloc((void **)&plan[i].d_gammay,sizeof(float)*nty);
		cudaMalloc((void **)&plan[i].d_alphay,sizeof(float)*nty);
		cudaMalloc((void **)&plan[i].d_Omegay,sizeof(float)*nty);
		cudaMalloc((void **)&plan[i].d_a_y,sizeof(float)*nty);
		cudaMalloc((void **)&plan[i].d_b_y,sizeof(float)*nty);
		cudaMalloc((void **)&plan[i].d_gammaz,sizeof(float)*ntz);
		cudaMalloc((void **)&plan[i].d_alphaz,sizeof(float)*ntz);
		cudaMalloc((void **)&plan[i].d_Omegaz,sizeof(float)*ntz);
		cudaMalloc((void **)&plan[i].d_a_z,sizeof(float)*ntz);
		cudaMalloc((void **)&plan[i].d_b_z,sizeof(float)*ntz);	
	
		cudaMalloc((void**)&plan[i].d_phi_vx_xx,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vy_yx,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vz_zx,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vx_xy,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vy_yy,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vz_zy,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vx_xz,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vy_yz,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vz_zz,size_model);
		
		cudaMalloc((void**)&plan[i].d_phi_vx_z,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vz_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vx_y,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vy_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vy_z,size_model);
		cudaMalloc((void**)&plan[i].d_phi_vz_y,size_model);

		cudaMalloc((void**)&plan[i].d_phi_pxx_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_pxy_y,size_model);
		cudaMalloc((void**)&plan[i].d_phi_pxz_z,size_model);
		cudaMalloc((void**)&plan[i].d_phi_pxy_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_pyy_y,size_model);
		cudaMalloc((void**)&plan[i].d_phi_pyz_z,size_model);
		cudaMalloc((void**)&plan[i].d_phi_pxz_x,size_model);
		cudaMalloc((void**)&plan[i].d_phi_pyz_y,size_model);
		cudaMalloc((void**)&plan[i].d_phi_pzz_z,size_model);
///////////////////////////////////////////////////////////////////////
		
		cudaMalloc((void **)&plan[i].d_inx, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_iny, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_inz, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_in_pxx, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_in_pyy, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_in_pzz, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_in_pxy, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_in_pxz, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_in_pyz, sizeof(cufftComplex)*ntp);

		cudaMalloc((void **)&plan[i].d_outx, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_outy, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_outz, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_outpxx, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_outpyy, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_outpzz, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_outpxy, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_outpxz, sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_outpyz, sizeof(cufftComplex)*ntp);
			
		cudaMalloc((void**)&plan[i].d_kx,sizeof(float)*ntx);
		cudaMalloc((void**)&plan[i].d_ky,sizeof(float)*nty);
		cudaMalloc((void**)&plan[i].d_kz,sizeof(float)*ntz);
		cudaMalloc((void**)&plan[i].d_k,size_model);	

		cudaMalloc((void **)&plan[i].d_kvx_x,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_kvy_y,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_kvz_z,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_kvx_z,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_kvz_x,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_kvx_y,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_kvy_x,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_kvy_z,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_kvz_y,sizeof(cufftComplex)*ntp);

		
//////////////////////////////////////////////////////////////
		cudaMalloc((void **)&plan[i].d_eta_p1,size_model);
		cudaMalloc((void **)&plan[i].d_eta_p2,size_model);
		cudaMalloc((void **)&plan[i].d_eta_p3,size_model);
		cudaMalloc((void **)&plan[i].d_eta_s1,size_model);
		cudaMalloc((void **)&plan[i].d_eta_s2,size_model);
		cudaMalloc((void **)&plan[i].d_eta_s3,size_model);
		cudaMalloc((void **)&plan[i].d_tao_p1,size_model);
		cudaMalloc((void **)&plan[i].d_tao_p2,size_model);
		cudaMalloc((void **)&plan[i].d_tao_s1,size_model);
		cudaMalloc((void **)&plan[i].d_tao_s2,size_model);
//////////////////////////////////////////////////////////////

////////////////////////////////////////
		cudaMalloc((void **)&plan[i].d_Ap1,size_model);
		cudaMalloc((void **)&plan[i].d_Ap2,size_model);
		cudaMalloc((void **)&plan[i].d_Ap3,size_model);
///////////////////////////////////////	

		cudaMalloc((void **)&plan[i].d_partx1,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_party1,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partz1,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partx2,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_party2,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partz2,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partx3,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_party3,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partz3,sizeof(cufftComplex)*ntp);

		cudaMalloc((void **)&plan[i].d_partvx_x1,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvx_x2,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvx_x3,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvx_x4,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvx_x5,sizeof(cufftComplex)*ntp);

		cudaMalloc((void **)&plan[i].d_partvy_y1,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvy_y2,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvy_y3,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvy_y4,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvy_y5,sizeof(cufftComplex)*ntp);

		cudaMalloc((void **)&plan[i].d_partvz_z1,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvz_z2,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvz_z3,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvz_z4,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvz_z5,sizeof(cufftComplex)*ntp);

		cudaMalloc((void **)&plan[i].d_partvx_y1,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvx_y2,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvx_y3,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvx_y4,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvx_y5,sizeof(cufftComplex)*ntp);

		cudaMalloc((void **)&plan[i].d_partvy_x1,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvy_x2,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvy_x3,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvy_x4,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvy_x5,sizeof(cufftComplex)*ntp);

		cudaMalloc((void **)&plan[i].d_partvz_x1,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvz_x2,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvz_x3,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvz_x4,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvz_x5,sizeof(cufftComplex)*ntp);

		cudaMalloc((void **)&plan[i].d_partvx_z1,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvx_z2,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvx_z3,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvx_z4,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvx_z5,sizeof(cufftComplex)*ntp);

		cudaMalloc((void **)&plan[i].d_partvy_z1,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvy_z2,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvy_z3,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvy_z4,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvy_z5,sizeof(cufftComplex)*ntp);

		cudaMalloc((void **)&plan[i].d_partvz_y1,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvz_y2,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvz_y3,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvz_y4,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_partvz_y5,sizeof(cufftComplex)*ntp);
	////////////	
		cudaMalloc((void **)&plan[i].d_record, sizeof(float)*rnmax*nt);	
		cudaMalloc((void **)&plan[i].d_record2, sizeof(float)*rnmax*nt);	
		cudaMalloc((void **)&plan[i].d_record3, sizeof(float)*rnmax*nt);		

		cudaMalloc((void **)&plan[i].d_dvx,sizeof(cufftComplex)*ntp);
		cudaMalloc((void **)&plan[i].d_dvy,sizeof(cufftComplex)*ntp);	
		cudaMalloc((void **)&plan[i].d_dvz,sizeof(cufftComplex)*ntp);		
	}
}


//=========================================================
//  Free the memory for variables in device
//  =======================================================
extern "C"
void cuda_Device_free
(
	struct MultiGPU plan[], int GPU_N
)
{
	int i;	 

	for(i=0;i<GPU_N;i++)
	{
		cudaSetDevice(i);

		cufftDestroy(plan[i].PLAN_FORWARD);
		cufftDestroy(plan[i].PLAN_BACKWARD);

		cudaFreeHost(plan[i].pxx);
		cudaFreeHost(plan[i].pyy);
		cudaFreeHost(plan[i].pzz);
		cudaFreeHost(plan[i].pxy);
		cudaFreeHost(plan[i].pxz);
		cudaFreeHost(plan[i].pyz);
		cudaFreeHost(plan[i].vx);
		cudaFreeHost(plan[i].vy);
		cudaFreeHost(plan[i].vz);

		cudaFreeHost(plan[i].record);
		cudaFreeHost(plan[i].record2);
		cudaFreeHost(plan[i].record3);

		

		cudaFree(plan[i].d_r_ix);
		cudaFree(plan[i].d_r_iy);
		cudaFree(plan[i].d_rik);

		cudaFree(plan[i].d_velp);
		cudaFree(plan[i].d_gama_p);
		cudaFree(plan[i].d_vels);
		cudaFree(plan[i].d_gama_s);
		cudaFree(plan[i].d_rho);

		cudaFree(plan[i].d_pxx);
		cudaFree(plan[i].d_pyy);
		cudaFree(plan[i].d_pzz);
		cudaFree(plan[i].d_pxy);
		cudaFree(plan[i].d_pxz);
		cudaFree(plan[i].d_pyz);
		cudaFree(plan[i].d_vx);
		cudaFree(plan[i].d_vy);
		cudaFree(plan[i].d_vz);

//////////////////pml /////////////////
		cudaFree(plan[i].d_gammax);
		cudaFree(plan[i].d_alphax);
		cudaFree(plan[i].d_Omegax);
		cudaFree(plan[i].d_a_x);
		cudaFree(plan[i].d_b_x);
		cudaFree(plan[i].d_gammay);
		cudaFree(plan[i].d_alphay);
		cudaFree(plan[i].d_Omegay);
		cudaFree(plan[i].d_a_y);
		cudaFree(plan[i].d_b_y);
		cudaFree(plan[i].d_gammaz);
		cudaFree(plan[i].d_alphaz);
		cudaFree(plan[i].d_Omegaz);
		cudaFree(plan[i].d_a_z);
		cudaFree(plan[i].d_b_z);

		cudaFree(plan[i].d_phi_vx_xx);
		cudaFree(plan[i].d_phi_vy_yx);
		cudaFree(plan[i].d_phi_vz_zx);
		cudaFree(plan[i].d_phi_vx_xy);
		cudaFree(plan[i].d_phi_vy_yy);
		cudaFree(plan[i].d_phi_vz_zy);
		cudaFree(plan[i].d_phi_vx_xz);
		cudaFree(plan[i].d_phi_vy_yz);
		cudaFree(plan[i].d_phi_vz_zz);
	
		cudaFree(plan[i].d_phi_vx_z);
		cudaFree(plan[i].d_phi_vz_x);
		cudaFree(plan[i].d_phi_vx_y);
		cudaFree(plan[i].d_phi_vy_x);
		cudaFree(plan[i].d_phi_vy_z);
		cudaFree(plan[i].d_phi_vz_y);

		cudaFree(plan[i].d_phi_pxx_x);
		cudaFree(plan[i].d_phi_pxy_y);
		cudaFree(plan[i].d_phi_pxz_z);
		cudaFree(plan[i].d_phi_pxy_x);
		cudaFree(plan[i].d_phi_pyy_y);
		cudaFree(plan[i].d_phi_pyz_z);
		cudaFree(plan[i].d_phi_pxz_x);
		cudaFree(plan[i].d_phi_pyz_y);
		cudaFree(plan[i].d_phi_pzz_z);
//////////////////////////////////////////////////////

		cudaFree(plan[i].d_inx);
		cudaFree(plan[i].d_iny);
		cudaFree(plan[i].d_inz);
		cudaFree(plan[i].d_in_pxx);
		cudaFree(plan[i].d_in_pyy);
		cudaFree(plan[i].d_in_pzz);
		cudaFree(plan[i].d_in_pxy);
		cudaFree(plan[i].d_in_pxz);
		cudaFree(plan[i].d_in_pyz);

		cudaFree(plan[i].d_outx);
		cudaFree(plan[i].d_outy);
		cudaFree(plan[i].d_outz);
		cudaFree(plan[i].d_outpxx);
		cudaFree(plan[i].d_outpyy);
		cudaFree(plan[i].d_outpzz);
		cudaFree(plan[i].d_outpxy);
		cudaFree(plan[i].d_outpxz);
		cudaFree(plan[i].d_outpyz);
		

		cudaFree(plan[i].d_kx);
		cudaFree(plan[i].d_ky);
		cudaFree(plan[i].d_kz);
		cudaFree(plan[i].d_k);

		cudaFree(plan[i].d_kvx_x);
		cudaFree(plan[i].d_kvy_y);
		cudaFree(plan[i].d_kvz_z);
		cudaFree(plan[i].d_kvx_z);
		cudaFree(plan[i].d_kvz_x);
		cudaFree(plan[i].d_kvy_z);
		cudaFree(plan[i].d_kvz_y);
		cudaFree(plan[i].d_kvx_y);
		cudaFree(plan[i].d_kvy_x);

		cudaFree(plan[i].d_eta_p1);
		cudaFree(plan[i].d_eta_p2);
		cudaFree(plan[i].d_eta_p3);
		cudaFree(plan[i].d_eta_s1);
		cudaFree(plan[i].d_eta_s2);
		cudaFree(plan[i].d_eta_s3);
		cudaFree(plan[i].d_tao_p1);
		cudaFree(plan[i].d_tao_p2);
		cudaFree(plan[i].d_tao_s1);
		cudaFree(plan[i].d_tao_s2);
		cudaFree(plan[i].d_Ap1);
		cudaFree(plan[i].d_Ap2);
		cudaFree(plan[i].d_Ap3);

		cudaFree(plan[i].d_partx1);
		cudaFree(plan[i].d_party1);
		cudaFree(plan[i].d_partz1);
		cudaFree(plan[i].d_partx2);
		cudaFree(plan[i].d_party2);
		cudaFree(plan[i].d_partz2);
		cudaFree(plan[i].d_partx3);
		cudaFree(plan[i].d_party3);
		cudaFree(plan[i].d_partz3);

		cudaFree(plan[i].d_partvx_x1);
		cudaFree(plan[i].d_partvx_x2);
		cudaFree(plan[i].d_partvx_x3);
		cudaFree(plan[i].d_partvx_x4);
		cudaFree(plan[i].d_partvx_x5);

		cudaFree(plan[i].d_partvy_y1);
		cudaFree(plan[i].d_partvy_y2);
		cudaFree(plan[i].d_partvy_y3);
		cudaFree(plan[i].d_partvy_y4);
		cudaFree(plan[i].d_partvy_y5);

		cudaFree(plan[i].d_partvz_z1);
		cudaFree(plan[i].d_partvz_z2);
		cudaFree(plan[i].d_partvz_z3);
		cudaFree(plan[i].d_partvz_z4);
		cudaFree(plan[i].d_partvz_z5);

		cudaFree(plan[i].d_partvx_y1);
		cudaFree(plan[i].d_partvx_y2);
		cudaFree(plan[i].d_partvx_y3);
		cudaFree(plan[i].d_partvx_y4);
		cudaFree(plan[i].d_partvx_y5);

		cudaFree(plan[i].d_partvy_x1);
		cudaFree(plan[i].d_partvy_x2);
		cudaFree(plan[i].d_partvy_x3);
		cudaFree(plan[i].d_partvy_x4);
		cudaFree(plan[i].d_partvy_x5);

		cudaFree(plan[i].d_partvx_z1);
		cudaFree(plan[i].d_partvx_z2);
		cudaFree(plan[i].d_partvx_z3);
		cudaFree(plan[i].d_partvx_z4);
		cudaFree(plan[i].d_partvx_z5);

		cudaFree(plan[i].d_partvz_x1);
		cudaFree(plan[i].d_partvz_x2);
		cudaFree(plan[i].d_partvz_x3);
		cudaFree(plan[i].d_partvz_x4);
		cudaFree(plan[i].d_partvz_x5);

		cudaFree(plan[i].d_partvy_z1);
		cudaFree(plan[i].d_partvy_z2);
		cudaFree(plan[i].d_partvy_z3);
		cudaFree(plan[i].d_partvy_z4);
		cudaFree(plan[i].d_partvy_z5);

		cudaFree(plan[i].d_partvz_y1);
		cudaFree(plan[i].d_partvz_y2);
		cudaFree(plan[i].d_partvz_y3);
		cudaFree(plan[i].d_partvz_y4);
		cudaFree(plan[i].d_partvz_y5);
		
		cudaFree(plan[i].d_record);
		cudaFree(plan[i].d_record2);
		cudaFree(plan[i].d_record3);


		cudaFree(plan[i].d_dvx);
		cudaFree(plan[i].d_dvy);
		cudaFree(plan[i].d_dvz);
	}
}


extern "C"
void getdevice(int *GPU_N)
{
	cudaGetDeviceCount(GPU_N);	
}
