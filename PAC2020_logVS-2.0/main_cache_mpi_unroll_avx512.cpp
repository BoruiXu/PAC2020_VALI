/* 
 * logDataVSPrior is a function to calculate 
 * the accumulation from ABS of two groups of complex data
 * *************************************************************************/

// using simd, data aligning, openmp ,fma and so on
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <omp.h>
#include <immintrin.h> 
#include <malloc.h>
#include <mpi.h>
using namespace std;

// typedef complex<double> Complex;
typedef chrono::high_resolution_clock Clock;

const int m=1638400;    // DO NOT CHANGE!!
const int K=100000; // DO NOT CHANGE!!
const int count = 24;
const int block = 480;
const int local_num = K/4;

inline void logDataVSPrior(const double* dat_0, const double* dat_1,const double* pri_0, const double* pri_1, const double* ctf, const double* sigRcp, const int num, const double disturb0,double *temp_data);





int main ( int argc, char *argv[] )
{ 
    double* final_data = new double[K];
    // memset(final_data,0,sizeof(final_data));

    int my_Rank,comm_sz;
    int source;
    double total_result;

    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);


    double *disturb = new double[K];


    double* dat_0 = (double*) memalign(32,m*8);
    double* dat_1 = (double*) memalign(32,m*8);
    double* pri_0 = (double*) memalign(32,m*8);
    double* pri_1 = (double*) memalign(32,m*8);
    double* ctf = (double*) memalign(32,m*8);
    double* sigRcp = (double*) memalign(32,m*8);

    double* temp_data = new double[local_num];
    memset(temp_data,0,sizeof(temp_data));

    //double* final_data = NULL;



    
    double dat0, dat1, pri0, pri1, ctf0, sigRcp0;

    

    /***************************
     * Read data from input.dat
     * *************************/
    ifstream fin;
    cout<<"start reading ..."<<endl;

    fin.open("input.dat");
    if(!fin.is_open())
    {
        cout << "Error opening file input.dat" << endl;
        exit(1);
    }
    int i=0;
    while( !fin.eof() ) 
    {
        fin >> dat0 >> dat1 >> pri0 >> pri1 >> ctf0 >> sigRcp0;
        dat_0[i]=dat0;
        dat_1[i]=dat1;
        pri_0[i]=pri0;
        pri_1[i]=pri1;
        ctf[i] = ctf0;
        sigRcp[i] = sigRcp0;
        i++;
        if(i == m) break;
    }
    fin.close();

    fin.open("K.dat");
    if(!fin.is_open())
    {
    cout << "Error opening file K.dat" << endl;
    exit(1);
    }
    i=0;
    while( !fin.eof() )
    {
    fin >> disturb[i];
    i++;
    if(i == K) break;
    }
    fin.close();

    cout<<"reading finished, start compting"<<endl;
    /***************************
     * main computation is here
     * ************************/
    auto startTime = Clock::now(); 
   // double result;

    ofstream fout;
    fout.open("result.dat");
    if(!fout.is_open())
    {
         cout << "Error opening file for result" << endl;
         exit(1);
    }
    int start = my_Rank*local_num;
    int end = start+local_num;


    #pragma omp parallel num_threads(count) 
    for(int j=0;j<m;j+=block){
     #pragma omp for 
        for(unsigned int t = start; t < end; t++)
        {

            logDataVSPrior(dat_0, dat_1,pri_0, pri_1,ctf, sigRcp, j, disturb[t],&temp_data[t%local_num]);

        }
    }


    if(my_Rank==0){

        MPI_Gather(temp_data,local_num,MPI_DOUBLE,final_data,local_num,MPI_DOUBLE,0,MPI_COMM_WORLD);

        for(int t=0;t<K;t++){
            fout << t+1 << ": " << final_data[t] <<"\n";
        }
   
        fout<<flush;

    }
    else{
        MPI_Gather(temp_data,local_num,MPI_DOUBLE,final_data,local_num,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }

    
    fout.close();

    auto endTime = Clock::now(); 

    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    if(my_Rank==0)
        cout << "Computing time=" << compTime.count() << " microseconds" << endl;


    free((void* )dat_0);
    free((void* )pri_0);
    free((void* )dat_1);
    free((void* )pri_1);
    free((void* )ctf);
    free((void* )sigRcp);

    delete[] final_data;
    delete[] disturb;
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}

inline void logDataVSPrior(const double* dat_0, const double* dat_1,const double* pri_0, const double* pri_1, const double* ctf, const double* sigRcp, const int num, const double disturb0,double* temp_data)
{
  
    int i;
    
    double my_result=0.0;
    double my_result_array[16];
    
    
    __m512d m_dat0, m_dat1, m_pri0, m_pri1,m_ctf,m_sigRcp,temp0,temp1,temp2,temp3,temp4; 
    __m512d m_disturb0 = _mm512_set_pd(disturb0,disturb0,disturb0,disturb0,disturb0,disturb0,disturb0,disturb0);

    __m512d local_my_result = _mm512_set_pd(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
    __m512d local_my_result2 = _mm512_set_pd(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
    for (i = num; i < num+block; i+=16)
    {   
	    m_ctf = _mm512_load_pd(ctf+i);
        m_dat0 = _mm512_load_pd(dat_0+i);
        m_pri0 = _mm512_load_pd(pri_0+i);
        m_dat1 = _mm512_load_pd(dat_1+i);
        m_pri1 = _mm512_load_pd(pri_1+i);
        m_sigRcp = _mm512_load_pd(sigRcp+i);


        temp0 = _mm512_mul_pd(m_ctf,m_disturb0);
        temp1 = _mm512_fmsub_pd(m_pri0,temp0,m_dat0);
        temp2 = _mm512_fmsub_pd(m_pri1,temp0,m_dat1);
        temp3 = _mm512_mul_pd(temp1,temp1);
        temp4 = _mm512_fmadd_pd(temp2,temp2,temp3);
        local_my_result = _mm512_fmadd_pd(temp4,m_sigRcp,local_my_result);

        
        m_ctf = _mm512_load_pd(ctf+i+8);//2
        m_dat0 = _mm512_load_pd(dat_0+i+8);//2
        m_pri0 = _mm512_load_pd(pri_0+i+8);//2
        m_dat1 = _mm512_load_pd(dat_1+i+8);//2
        m_pri1 = _mm512_load_pd(pri_1+i+8);//2
        m_sigRcp = _mm512_load_pd(sigRcp+i+8);//2


        temp0 = _mm512_mul_pd(m_ctf,m_disturb0);//2
        temp1 = _mm512_fmsub_pd(m_pri0,temp0,m_dat0);//2
        temp2 = _mm512_fmsub_pd(m_pri1,temp0,m_dat1);//2
        temp3 = _mm512_mul_pd(temp1,temp1);//2
        temp4 = _mm512_fmadd_pd(temp2,temp2,temp3);//2
        local_my_result2 = _mm512_fmadd_pd(temp4,m_sigRcp,local_my_result2);//2
    
    }
   // local_my_result = _mm256_add_pd(local_my_result,local_my_result2);

    _mm512_storeu_pd(my_result_array,local_my_result);
   _mm512_storeu_pd(&my_result_array[8],local_my_result2);
   
    my_result=(my_result_array[0]+my_result_array[1])+(my_result_array[2]+my_result_array[3])+(my_result_array[4]+my_result_array[5])+(my_result_array[6]+my_result_array[7])+
    (my_result_array[8]+my_result_array[9])+(my_result_array[10]+my_result_array[11])+(my_result_array[12]+my_result_array[13])+(my_result_array[14]+my_result_array[15]);


    
    
    
    *temp_data+=my_result;


}




