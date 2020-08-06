/* 
 * logDataVSPrior is a function to calculate 
 * the accumulation from ABS of two groups of complex data
 * *************************************************************************/

// using simd, data aligning, openmp ,fma, cache blocking mpi and so on 
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
const int count = 80;
const int block = 400;
const int local_num = K/4;

inline void logDataVSPrior(const double* dat_0, const double* dat_1,const double* pri_0, const double* pri_1, const double* ctf, const double* sigRcp, const int num, const double disturb0,double *temp_data);





int main ( int argc, char *argv[] )
{ 
    double* final_data = new double[K];
    // memset(final_data,0,sizeof(final_data));

    int my_Rank,comm_sz;
    
    double total_result;

    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);


    //double *disturb = new double[K];


    double* dat_0 = (double*) memalign(32,m*8);
    double* dat_1 = (double*) memalign(32,m*8);
    double* pri_0 = (double*) memalign(32,m*8);
    double* pri_1 = (double*) memalign(32,m*8);
    double* ctf = (double*) memalign(32,m*8);
    double* sigRcp = (double*) memalign(32,m*8);

    double* temp_data = new double[local_num];
    memset(temp_data,0,sizeof(temp_data));

    



    
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
    double my_result_array[4];
    
    
    __m256d m_dat0, m_dat1, m_pri0, m_pri1,m_ctf,m_sigRcp; 
    __m256d m_disturb0 = _mm256_set_pd(disturb0,disturb0,disturb0,disturb0);

    __m256d local_my_result = _mm256_set_pd(0.0,0.0,0.0,0.0);
    for (i = num; i < num+block; i+=4)
    {   

        m_dat0 = _mm256_load_pd(dat_0+i);
        m_dat1 = _mm256_load_pd(dat_1+i);
        m_pri0 = _mm256_load_pd(pri_0+i);
        m_pri1 = _mm256_load_pd(pri_1+i);
        m_ctf = _mm256_load_pd(ctf+i);
        m_sigRcp = _mm256_load_pd(sigRcp+i);

       
        m_ctf = _mm256_mul_pd(m_ctf,m_disturb0);
        m_dat0 = _mm256_fmsub_pd(m_pri0,m_ctf,m_dat0);
        m_dat1 = _mm256_fmsub_pd(m_pri1,m_ctf,m_dat1);
        m_dat0 = _mm256_mul_pd(m_dat0,m_dat0);
        m_dat0 = _mm256_fmadd_pd(m_dat1,m_dat1,m_dat0);

    
        local_my_result = _mm256_fmadd_pd(m_dat0,m_sigRcp,local_my_result);
    
    }

    _mm256_storeu_pd(my_result_array,local_my_result);



    
    my_result=(my_result_array[0]+my_result_array[1])+(my_result_array[2]+my_result_array[3]);
    
    
    *temp_data+=my_result;


}




