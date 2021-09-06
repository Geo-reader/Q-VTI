//========3D viscoacoustic reverse time migration===//
//========3D viscoacoustic reverse time migration===//
//========3D viscoacoustic reverse time migration===//

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "mpi.h"
#include <string.h>
#include "Myfunctions.h"
using namespace std;

#define pi 3.1415926

int main(int argc, char* argv[])
{
  	//============MPI index====================//
  	//============MPI index====================//
  	//============MPI index====================//

	int myid,numprocs,namelen;
	
	MPI_Comm comm=MPI_COMM_WORLD;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(comm,&myid);
	MPI_Comm_size(comm,&numprocs);
	MPI_Get_processor_name(processor_name, &namelen);
	
	if(myid==0)
	{
		printf("\n==============================================\n");	
		printf("Number of MPI thread is %d\n",numprocs);
		printf("\n==============================================\n");	
	}

	//===================Flags definition======================//
	//===================Flags definition======================//

	int RTM_type=1;	// RTM_type=1, Q-compensated RTM
									// RTM_type=2, RTM using viscoelastic data, (without compensation)
									// RTM_type=3, RTM in lossless media (reference RTM)
		
	int imaging_condition=1;		// imaging_condition=1, Excitation amplitude imaging condition (EXA)
									// imaging_condition=2, Partial correlation (stable compensation)
									// imaging_condition=3, Source-normalized cross-correlation imaging condition
									// imaging_condition=4, cross-correlation imaging condition

	int forward_or_not=1;				//forward_or_not=1, forkward to get seismic record, =0 for not.
	
	int direcr_or_not=0;				//direcr_or_not=1, forkward to get direct wave, =0 for not.
	
	
	int media_type;	// media=1;  viscoacoustic
								// media=2;  dispersion-only
								// media=3;  acoustic (lossless)
								// media=4;  Q-compensation (amplitude amplifying strategy)
									
	int velocity_type;			// velocity_type=1 for homogeneous model
											// velocity_type=2 for true model
											// velocity_type=3 for smooth model
									
	int reconstruction;			//reconstruction=1 for executing wavefield reconstruction, reconstruction=0 for not.


  	//============Time begin====================//
  	//============Time begin====================//
	clock_t start,finish;
	if(myid==0)
	{
		start=clock();	
	}
	
	time_t begin_time;	
	time(&begin_time);
		

	//==============model parameters=============//
	//==============model parameters=============//
	int nx=505;
	int ny=455;
	int nz=19;
  	int rnmax=3281;				// maxium receiver numbers
	int pml=15;
	int nsx=1;
	int nsy=1;
	int ns=nsx*nsy;
	int nt=3000;
	int dsx=30;
	int dsy=30;
	int nsx0=nx/2;				//the distance of first shot to the left side of the model
	int nsy0=ny/2;				//the distance of first shot to the left side of the model
	int depth=1;			//shot and receiver depth
  	int dr=1;								//receivers distributed in every two points in both X- and Y-directions

	int i,GPU_N;						// GPU_N stands for the total number of GPUs per node
	
	getdevice(&GPU_N);					// Obtain the number of GPUs per node
	GPU_N=1;
	
	
	float dx=20.0;
	float dy=20.0;
	float dz=20.0;
	float dt=0.0008;
	float f0=20.0;
	float w0=2*pi*f0;			//reference frequency for viscoelastic extrapolation
	
	float taper_ratio=0.2;	// taper ratio for turkey window filter	
	float fcut=100.0;			// cut-off frequency over the maximum P-wave velocity
	
	int it,ix,iz,iy,ir, ip,ipp;	
	int ntz=nz+2*pml;			// total vertical samples 
	int nty=ny+2*pml;			// total vertical samples 
	int ntx=nx+2*pml;			// total horizontal samples 
	
	int nxh=ntx/2;			// half of vertical samples
	int nyh=nty/2;			// half of vertical samples
	int nzh=ntz/2;			// half of horizontal samples
	
	int ntp=ntz*ntx*nty;		// total samples of the model
	int np=nx*nz*ny;			// total samples of the simulating domain
	
	char filename[80];		// filename char
	FILE *fp;
	
	if(myid==0)
	{
		printf("ntx=%d, nty=%d, ntz=%d\n",ntx, nty, ntz);		
		printf("nx=%d,  ny=%d,  nz=%d,  nt=%d,  ns=%d\n",nx, ny, nz, nt, ns);		
		printf("dx=%f,  dy=%f,  dz=%f,  dt=%f\n",dx, dy, dz, dt);		
		printf("==============================================\n");
	}

	//================wavelet definition============//
	//================wavelet definition============//
	float *rik;			// ricker wavelet
	rik=(float *) malloc(nt*sizeof(float));
	
	getrik(nt, dt, f0, rik);
	
	//==============observation system definition===============//
	//==============observation system definition===============//
  	int is, isx, isy;  	
  	rnmax=0;						// maxium receiver numbers
  	
  	int rnx_max=nx/dr;			//maximum receiver number along X-direction
  	int rny_max=ny/dr;			//maximum receiver number along Y-direction
  	
	struct Source ss[ns];			// struct pointer for source variables

	for(isx=0; isx<nsx; isx++)
	{
		for(isy=0; isy<nsy; isy++)
		{
			is=isx*nsy+isy;
			
			ss[is].s_ix=isx*dsx+pml+nsx0;	// receiver horizontal position -X
			ss[is].s_iy=isy*dsy+pml+nsy0;	// receiver horizontal position -Y
			
			ss[is].s_iz=pml+depth;			// shot vertical position
			ss[is].r_iz=pml+depth;			// receiver vertical position
			
			ss[is].r_n =rnx_max*rny_max; 	
						
			if(rnmax<ss[is].r_n)
				rnmax=ss[is].r_n;		// maxium receiver numbers
		}
	}

	for(is=0;is<ns;is++)
	{
		ss[is].r_ix=(int*)malloc(sizeof(int)*rnmax);
		ss[is].r_iy=(int*)malloc(sizeof(int)*rnmax);
	}

	for(is=0;is<ns;is++)
	{
		for(ip=0;ip<rnmax;ip++)
		{
			ss[is].r_ix[ip] = pml+ip/(rny_max)*dr;
			ss[is].r_iy[ip] = pml+ip%(rny_max)*dr;
		}
	}

	fp=fopen("./output/source_xy.txt","wt");
	for(is=0;is<ns;is++)
	{
		fprintf(fp,"%10d %10d\r\n",ss[is].s_ix-pml,ss[is].s_iy-pml);
	}
	fclose(fp);
	fp=fopen("./output/source_trnum.txt","wt");
	for(is=0;is<ns;is++)
	{
		fprintf(fp,"%d\r\n",ss[is].r_n);
	}
	fclose(fp);

	/*for(is=0;is<ns;is++)
	{
		sprintf(filename,"./output/source%d_tracexy.txt",is+1);
		fp=fopen(filename,"wt");
		for(ix=0;ix<ss[is].r_n;ix++)
		{
			fprintf(fp,"%10d %10d\r\n",ss[is].r_ix[ix]-L,ss[is].r_iy[ix]-L);
		}
		fclose(fp);
	}*/

	if(myid==0)
	{
		printf("The total shot number is %d\n",ns);
		printf("The maximum trace number for source is %d\n",rnmax);
		printf("\n==============================================\n");	
	}


	//=========================================================
	//  Parameters of checkpoints... checkpoints distribute averagely
	//  =======================================================
	int check_steps;			// checkpoints interval
	int N_cp;					// total number of checkpoints
	int *t_cp;					// time point for checkpoints

	check_steps=500;							// every 200 timestep has a checkpoint
	N_cp= (int)(nt/check_steps);				// total number of checkpoints (in pair)
	t_cp= (int *) malloc(N_cp*sizeof(int));		// checkpoints pointer


	//==================================================//
	//  Parameters of GPU...(we assume each node has the same number of GPUs)    //
	//  =================================================//

	int nsid,modsr,prcs;
	int iss,eachsid,offsets;	
	
	printf("The available Device number is %d on %s\n",GPU_N,processor_name);
	printf("\n==============================================\n");

	struct MultiGPU plan[GPU_N];		// struct pointer for MultiGPU variables
	
	nsid=ns/(GPU_N*numprocs);			// shots number per GPU card
	modsr=ns%(GPU_N*numprocs);			// residual shots after average shot distribution
	prcs=modsr/GPU_N;					// which GPUs at each node will have one more shot
	if(myid<prcs)
	{
		eachsid=nsid+1;					// if thread ID less than prcs, the corresponding GUPs have one more shot
		offsets=myid*(nsid+1)*GPU_N;	// the offset of the shots
	}
	else
	{
		eachsid=nsid;												// the rest GUPs have nsid shots
		offsets=prcs*(nsid+1)*GPU_N+(myid-prcs)*nsid*GPU_N;			// the offset of the shots (including previous shots)
	}


	//===================velocity and Q models=============//
	//===================velocity and Q models=============//
	float *velp, *Qp, *gama_p, *vels, *Qs, *gama_s, *rho;
	float *c_velp, *c_Qp, *c_gama_p;
	
	velp = (float*)malloc(sizeof(float)*ntp);
	Qp = (float*)malloc(sizeof(float)*ntp);
	gama_p = (float*)malloc(sizeof(float)*ntp);
	vels = (float*)malloc(sizeof(float)*ntp);
	Qs = (float*)malloc(sizeof(float)*ntp);
	gama_s = (float*)malloc(sizeof(float)*ntp);
	rho = (float*)malloc(sizeof(float)*ntp);

	c_velp = (float*)malloc(sizeof(float)*ntp);
	c_Qp = (float*)malloc(sizeof(float)*ntp);
	c_gama_p = (float*)malloc(sizeof(float)*ntp);
	
	float velp_max, Qp_max;
	float velp_min, Qp_min;
	float gamaaverage_p;

	if(myid==0)
	{
		get_velQ(pml, ntx, nty, ntz, ntp,nx, ny, nz, velp, Qp, vels, Qs, rho);	

		gamaaverage_p=0.0;	
		for(ip=0;ip<ntp;ip++)
		{
			gama_p[ip]=atan(1.0/Qp[ip])/pi;
			gama_s[ip]=atan(1.0/Qs[ip])/pi;
			gamaaverage_p+=gama_p[ip];
		}
		gamaaverage_p/=ntp;

		velp_max=0.0; Qp_max=0.0; 
		velp_min=6000.0; Qp_min=6000.0;
		
		for(ip=0;ip<ntp;ip++)
		{
			if(velp[ip]>=velp_max){velp_max=velp[ip];}
			if(Qp[ip]>=Qp_max){Qp_max=Qp[ip];}
			if(velp[ip]<=velp_min){velp_min=velp[ip];}
			if(Qp[ip]<=Qp_min){Qp_min=Qp[ip];}
		}
		
		printf("velp_max = %f\n",velp_max);
		printf("Qp_max = %f\n",Qp_max);
		printf("velp_min = %f\n",velp_min); 
		printf("Qp_min = %f\n",Qp_min);
		printf("\n");
		
		
		printf("gamaaverage_p=%f\n",gamaaverage_p);
		printf("==============================================\n");	
	}
	
	
	MPI_Bcast(velp, ntp, MPI_FLOAT, 0, comm);	
	MPI_Bcast(Qp, ntp, MPI_FLOAT, 0, comm);
	MPI_Bcast(gama_p, ntp, MPI_FLOAT, 0, comm);
	MPI_Bcast(vels, ntp, MPI_FLOAT, 0, comm);	
	MPI_Bcast(Qs, ntp, MPI_FLOAT, 0, comm);
	MPI_Bcast(gama_s, ntp, MPI_FLOAT, 0, comm);
	MPI_Bcast(rho, ntp, MPI_FLOAT, 0, comm);
	
	MPI_Bcast(&velp_max, 1, MPI_FLOAT, 0, comm);
	MPI_Bcast(&velp_min, 1, MPI_FLOAT, 0, comm);
	MPI_Bcast(&Qp_max, 1, MPI_FLOAT, 0, comm);
	MPI_Bcast(&Qp_min, 1, MPI_FLOAT, 0, comm);
	MPI_Bcast(&gamaaverage_p,1,MPI_FLOAT,0,comm);
	
	MPI_Barrier(comm);

	
	//=========================================================
	// The following two functions are responsible for alloc 
	// and initialization struct varibale plan for GPU_N
	//  =======================================================
	cuda_Device_malloc(ntx, nty, ntz, ntp, nx, ny, nz, nt, rnmax, plan, GPU_N);
	cuda_Host_initialization(ntp,nt, rnmax, plan, GPU_N);

	MPI_Barrier(comm);	// MPI barrier to ensure that all variables have been well-defined	

	if(myid==0)
	{
		printf("\n==============================================\n");	
		printf("forward begin!\n");			
		printf("\n==============================================\n");	
	}
	
	//=================forward modeling=================//
	//=================forward modeling=================//
	//=================forward modeling=================//

	if(forward_or_not==1)
	{
		for(iss=0; iss<eachsid; iss++)	// each GPU card compute eachside shots 
		{
			//============forward of homogeneous model==========//
			//============forward of homogeneous model==========//　

		/*	if(direcr_or_not==1)
			{
				is=offsets+iss*GPU_N;		// current shot index
		
				if(RTM_type==1 || RTM_type==2)
				{
					media_type=1;
				}

				if(RTM_type==3)
				{
					media_type=3;
				}
		
				velocity_type=1;
			
				reconstruction=0;		
		
				for(i=0;i<GPU_N;i++)
				{
					get_homegeneous_velQ(is, ntx, nty, ntz, velp, gama_p,
										c_velp, c_gama_p, ss[is+i].s_ix, ss[is+i].s_iy, ss[is+i].s_iz);
				}
			
				cuda_forward_acoustic_3D
				(
					imaging_condition, RTM_type, reconstruction, velocity_type, media_type, 
					myid, is, N_cp, t_cp,
					nt, ntx, nty, ntz, ntp, nx, ny, nz, pml, dx, dy, dz, dt, f0, w0, 
					velp_max, gamaaverage_p, rik, c_velp, c_gama_p, ss, plan, GPU_N, rnmax, rnx_max, rny_max,dr
				);
			}*/
		
			//============forward of heterogeneous model==========//
			//============forward of heterogeneous model==========//
			is=offsets+iss*GPU_N;		// current shot index


			cuda_forward_acoustic_3D
			(
				myid, is, 
				nt, ntx, nty, ntz, ntp, nx, ny, nz, pml, 
				dx, dy, dz, dt, f0, w0, velp_max,
				rik, velp, gama_p, vels, gama_s, rho,
				ss, plan, GPU_N, rnmax, rnx_max, rny_max,dr
			);		
	

			for(i=0;i<GPU_N;i++)
			{
			/*	for(ip=0;ip<rnmax*nt;ip++)
				{		
					plan[i].record[ip]=plan[i].record[ip]-plan[i].record_dir[ip];		
				}*/

		
					sprintf(filename,"./record/Q1dep_vx_small_shot%d.bin",is+i);
					fp=fopen(filename,"wb");
					for(ip=0;ip<rnmax;ip++)
							for(it=0;it<nt;it++)
								fwrite(&plan[i].record[ip*nt+it],sizeof(float),1,fp);
					fclose(fp);

					sprintf(filename,"./record/Q1dep_vy_small_shot%d.bin",is+i);
					fp=fopen(filename,"wb");
					for(ip=0;ip<rnmax;ip++)
							for(it=0;it<nt;it++)
								fwrite(&plan[i].record2[ip*nt+it],sizeof(float),1,fp);
					fclose(fp);

					sprintf(filename,"./record/Q1dep_vz_small_shot%d.bin",is+i);
					fp=fopen(filename,"wb");
					for(ip=0;ip<rnmax;ip++)
							for(it=0;it<nt;it++)
								fwrite(&plan[i].record3[ip*nt+it],sizeof(float),1,fp);
					fclose(fp);
	
			}
		}	
	}
	
	
	if(myid==0)
	{
		printf("\n==============================================\n");	
		printf("forward end!\n");	
		printf("\n==============================================\n");	
	}

	//===============free variables=====================//
	//===============free variables=====================//
	
	cuda_Device_free(plan, GPU_N);
	
	for(is=0;is<ns;is++){free(ss[is].r_ix);} 
	for(is=0;is<ns;is++){free(ss[is].r_iy);} 
	
//	free(t_cp);
	free(rik);
	free(velp);
	free(gama_p);
	free(Qp);

	free(vels);
	free(gama_s);
	free(Qs);
	free(rho);

	free(c_velp);
	free(c_gama_p);
	free(c_Qp);		

	finish=clock();
	float time;	
	time=(float)(finish-start)/CLOCKS_PER_SEC;	
	
	if(myid==0)
	{	
		printf("====================================\n");		
		printf("The running time is %fseconds\n",time);	
		printf("====================================\n");		
		printf("\n");	
		printf("The program has been finished\n");	
		printf("====================================\n");		
		printf("\n");	
	}
	
	MPI_Barrier(comm);
	MPI_Finalize();	
	
	return 0;
}


void getrik(int nt, float dt, float f0, float *rik)
{
	int it;	float tmp;

	for(it=0;it<nt;it++)
	{
		tmp=pow(pi*f0*(it*dt-1/f0),2.0);
		rik[it]=(float)((1.0-2.0*tmp)*exp(-tmp));		
	}

/*	
	float *a;
	float max=0.0;
	a=(float *) malloc(sizeof(float)*nt);
	
	for(it=1;it<=nt;it++)
	{
		tmp=pow(pi*f0*(it*dt-1.0/f0),2.0);
		a[it-1]=(float)((1.0-2.0*tmp)*exp(-tmp));		
	}
	
	for(it=1;it<nt;it++)
	{
		rik[it]=(a[it]-a[it-1])/dt;
		if(fabs(max)<fabs(rik[it]))
		{
			max=fabs(rik[it]);
		}
	}
	
	for(it=0;it<nt;it++)
	{
		rik[it]=rik[it]/max;
	}	
	
	FILE *fprik=fopen("rik.bin","wb");
	fwrite(rik,sizeof(float),nt,fprik);
	fclose(fprik);	
	free(a);
	*/
}


void get_velQ(int pml, int ntx, int nty, int ntz,int ntp, int nx, int ny, int nz, float *velp, float *Qp, float *vels, float *Qs, float *rho)
{
	int ix, iy, iz,ip,ip0;

	FILE *fp1=fopen("./input/vpnew19_455_505.bin","rb");
	for(ix=pml;ix<ntx-pml;ix++)
		for(iy=pml;iy<nty-pml;iy++)
			for(iz=pml;iz<ntz-pml;iz++)
					fread(&velp[ix*nty*ntz+iy*ntz+iz],sizeof(float),1,fp1);
	fclose(fp1);
	
/*	fp1=fopen("./input/Q_200_200_76.bin","rb");
	for(ix=pml;ix<ntx-pml;ix++)
		for(iy=pml;iy<nty-pml;iy++)
			for(iz=pml;iz<ntz-pml;iz++)
					fread(&Qs[ix*nty*ntz+iy*ntz+iz],sizeof(float),1,fp1);
	fclose(fp1);

	fp1=fopen("./input/overth_200_200_76vs.dat","rb");
	for(ix=pml;ix<ntx-pml;ix++)
		for(iy=pml;iy<nty-pml;iy++)
			for(iz=pml;iz<ntz-pml;iz++)
					fread(&vels[ix*nty*ntz+iy*ntz+iz],sizeof(float),1,fp1);
	fclose(fp1);
	
	fp1=fopen("./input/Q_200_200_76.bin","rb");
	for(ix=pml;ix<ntx-pml;ix++)
		for(iy=pml;iy<nty-pml;iy++)
			for(iz=pml;iz<ntz-pml;iz++)
					fread(&Qp[ix*nty*ntz+iy*ntz+iz],sizeof(float),1,fp1);
	fclose(fp1);

	fp1=fopen("./input/rho_200_200_76.bin","rb");
	for(ix=pml;ix<ntx-pml;ix++)
		for(iy=pml;iy<nty-pml;iy++)
			for(iz=pml;iz<ntz-pml;iz++)
					fread(&rho[ix*nty*ntz+iy*ntz+iz],sizeof(float),1,fp1);
	fclose(fp1);*/

    for(ix=pml;ix<ntx-pml;ix++)
    {
    	for(iy=pml; iy<nty-pml; iy++)
    	{
 		    for(iz=0;iz<pml;iz++)
		    {
		    	ip=ix*nty*ntz+iy*ntz+iz;
		    	ip0=ix*nty*ntz+iy*ntz+pml;
		        velp[ip]=velp[ip0];                 
		      /*  Qp[ip]=Qp[ip0];   
		        vels[ip]=vels[ip0];                 
		        Qs[ip]=Qs[ip0];  
		        rho[ip]=rho[ip0]; */ 
		    }  //top
		    for(iz=ntz-pml;iz<ntz;iz++)
		    {
		    	ip=ix*nty*ntz+iy*ntz+iz;
		    	ip0=ix*nty*ntz+iy*ntz+(ntz-pml-1);
		        velp[ip]=velp[ip0];                 
		     /*   Qp[ip]=Qp[ip0];   
		        vels[ip]=vels[ip0];                 
		        Qs[ip]=Qs[ip0];  
		        rho[ip]=rho[ip0]; */                  
		    }
    	}
    }  //bottom
    
    for(iz=0;iz<ntz;iz++)
    {
        for(ix=pml;ix<ntx-pml;ix++)
        {
        	for(iy=0; iy<pml; iy++)
        	{
		    	ip=ix*nty*ntz+iy*ntz+iz;
		    	ip0=ix*nty*ntz+pml*ntz+iz;
		        velp[ip]=velp[ip0];                 
		      /*  Qp[ip]=Qp[ip0];   
		        vels[ip]=vels[ip0];                 
		        Qs[ip]=Qs[ip0];  
		        rho[ip]=rho[ip0];  */                  
        	}	//left
        	for(iy=nty-pml; iy<nty; iy++)
        	{
		    	ip=ix*nty*ntz+iy*ntz+iz;
		    	ip0=ix*nty*ntz+(nty-pml-1)*ntz+iz;
		        velp[ip]=velp[ip0];                 
		    /*    Qp[ip]=Qp[ip0];   
		        vels[ip]=vels[ip0];                 
		        Qs[ip]=Qs[ip0];  
		        rho[ip]=rho[ip0];   */                 
        	}        	//right
        }
    }
    
    for(iz=0;iz<ntz;iz++)
    {
        for(iy=0;iy<nty;iy++)
        {
        	for(ix=0; ix<pml; ix++)
        	{
		    	ip=ix*nty*ntz+iy*ntz+iz;
		    	ip0=pml*nty*ntz+iy*ntz+iz;
		        velp[ip]=velp[ip0];                 
		    /*    Qp[ip]=Qp[ip0];   
		        vels[ip]=vels[ip0];                 
		        Qs[ip]=Qs[ip0];  
		        rho[ip]=rho[ip0];   */                 	
        	}        	//ahead
        	for(ix=ntx-pml; ix<ntx; ix++)
        	{
		    	ip=ix*nty*ntz+iy*ntz+iz;
		    	ip0=(ntx-pml-1)*nty*ntz+iy*ntz+iz;
		        velp[ip]=velp[ip0];                 
		     /*   Qp[ip]=Qp[ip0];   
		        vels[ip]=vels[ip0];                 
		        Qs[ip]=Qs[ip0];  
		        rho[ip]=rho[ip0];   */      
        	}        	//behind
        }
    }
	

	for(ix=0;ix<ntx;ix++)
	{
		for(iy=0;iy<nty;iy++)
		{
			for(iz=0;iz<ntz;iz++)
			{
		    	ip=ix*nty*ntz+iy*ntz+iz;
				///////velp[ip]=2500.0;
				
				vels[ip]=velp[ip]/1.5;
				Qp[ip]=200.0;
				Qs[ip]=150.0;

		//		Qp[ip]=300.0;
		//		Qs[ip]=190.0;
				rho[ip]=1800;
			}

		}
	}
/*    
	fp1=fopen("./input/velp1.bin","wb");
	for(ix=0;ix<ntx;ix++)
		for(iy=0;iy<nty;iy++)
			for(iz=0;iz<ntz;iz++)
					fwrite(&velp[ix*nty*ntz+iy*ntz+iz],sizeof(float),1,fp1);
	fclose(fp1);
	
	fp1=fopen("./input/Qp1.bin","wb");
	for(ix=0;ix<ntx;ix++)
		for(iy=0;iy<nty;iy++)
			for(iz=0;iz<ntz;iz++)
					fwrite(&Qp[ix*nty*ntz+iy*ntz+iz],sizeof(float),1,fp1);
	fclose(fp1);    
*/
}

void get_homegeneous_velQ
(
	int is, int ntx, int nty, int ntz, float *velp, float *gama_p,
	float *c_velp, float *c_gama_p, int s_ix, int s_iy, int s_iz
)
{
	int ix, iy, iz, ip, ip0;
	
	for(iz=0;iz<ntz;iz++)
	{
		for(iy=0;iy<nty; iy++)
		{
			for(ix=0;ix<ntx;ix++)
			{
		    	ip=ix*nty*ntz+iy*ntz+iz;
		    	ip0=s_ix*nty*ntz+s_iy*ntz+s_iz;
		    	
		        c_velp[ip]=velp[ip0];    
		        c_gama_p[ip]=gama_p[ip0];
			}
		}
	}
}

