/* 
 * logDataVSPrior is a function to calculate 
 * the accumulation from ABS of two groups of complex data
 * *************************************************************************/

// based on version1 and using struct
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <omp.h>
#include <immintrin.h> 
#include <malloc.h>
using namespace std;

// typedef complex<double> Complex;
typedef chrono::high_resolution_clock Clock;

const int m=1638400;    // DO NOT CHANGE!!
const int K=100000; // DO NOT CHANGE!!
const int count = 80;
const int len_num = m/4;
const int local_num = m/count;




struct group{
    
    double dat_0[4];
    double dat_1[4];
    double pri_0[4];
    double pri_1[4];
    double ctf[4];
    double sigRcp[4];

};

inline double logDataVSPrior(const group* g, const int num, const double disturb0);





int main ( int argc, char *argv[] )
{ 
    // Complex *dat = new Complex[m];
    // Complex *pri = new Complex[m];
    // double *dat_0 = new double[m];
    // double *dat_1 = new double[m];
    // double *pri_0 = new double[m];
    // double *pri_1 = new double[m];
    // double *ctf = new double[m];
    // double *sigRcp = new double[m];
    double *disturb = new double[K];


    group* g = new group[len_num];

    // double* dat_0 = (double*) memalign(32,m*8);
    // double* dat_1 = (double*) memalign(32,m*8);
    // double* pri_0 = (double*) memalign(32,m*8);
    // double* pri_1 = (double*) memalign(32,m*8);
    // double* ctf = (double*) memalign(32,m*8);
    // double* sigRcp = (double*) memalign(32,m*8);




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
    int j=0;
    while( !fin.eof() ) 
    {
        fin >> dat0 >> dat1 >> pri0 >> pri1 >> ctf0 >> sigRcp0;
        // dat[i] = Complex (dat0, dat1);
        // pri[i] = Complex (pri0, pri1);
        g[i].dat_0[j]=dat0;
        g[i].dat_1[j]=dat1;
        g[i].pri_0[j]=-pri0;
        g[i].pri_1[j]=-pri1;
        g[i].ctf[j] = ctf0;
        g[i].sigRcp[j] = sigRcp0;
        j++;
        if(j==4){
            i++;
            j=0;
        }
        
        if(i == len_num) break;
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
    double result;

    ofstream fout;
    fout.open("result.dat");
    if(!fout.is_open())
    {
         cout << "Error opening file for result" << endl;
         exit(1);
    }


    for(unsigned int t = 0; t < K; t++)
    {      
        result = logDataVSPrior(g, m, disturb[t]);
        fout << t+1 << ": " << result <<"\n";
    }
    fout<<flush;
    fout.close();

    auto endTime = Clock::now(); 

    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Computing time=" << compTime.count() << " microseconds" << endl;

    // delete[] dat_0;
    // delete[] pri_0;

    // delete[] dat_1;
    // delete[] pri_1;

    // delete[] ctf;
    // delete[] sigRcp;

    // free((void* )dat_0);
    // free((void* )pri_0);
    // free((void* )dat_1);
    // free((void* )pri_1);
    // free((void* )ctf);
    // free((void* )sigRcp);
    delete[] g;

    delete[] disturb;
    return EXIT_SUCCESS;
}

inline double logDataVSPrior(const group* g, const int num, const double disturb0)
{
    double result = 0.0;
    #pragma omp parallel num_threads(count)
  
{

       
    int my_rank = omp_get_thread_num();
    //int thread_count = omp_get_num_threads();
    int i;
    int start,end; 
    double my_result=0.0;
    //double *local_my_result = new double[local_num];
    double my_result_array[4];



    start = local_num*my_rank/4;
    end = start+local_num/4;
    
    
    __m256d m_dat0, m_dat1, m_pri0, m_pri1,m_ctf,m_sigRcp; 
    __m256d m_disturb0 = _mm256_set_pd(disturb0,disturb0,disturb0,disturb0);

    __m256d local_my_result = _mm256_set_pd(0.0,0.0,0.0,0.0);
    for (i = start; i < end; i++)
    {   

        m_dat0 = _mm256_loadu_pd(g[i].dat_0);
        m_dat1 = _mm256_loadu_pd(g[i].dat_1);
        m_pri0 = _mm256_loadu_pd(g[i].pri_0);
        m_pri1 = _mm256_loadu_pd(g[i].pri_1);
        m_ctf = _mm256_loadu_pd(g[i].ctf);
        m_sigRcp = _mm256_loadu_pd(g[i].sigRcp);

            //my_result+= ( norm( dat[i] - ctf[i] * pri[i] ) * sigRcp[i] );
        // m1 = _mm256_mul_pd(m_pri0,m_ctf);
        // m2 = _mm256_mul_pd(m_pri1,m_ctf);
        // m1 = _mm256_sub_pd(m_dat0,m1);
        // m2 = _mm256_sub_pd(m_dat1,m2);
        m_ctf = _mm256_mul_pd(m_ctf,m_disturb0);
        m_dat0 = _mm256_fmadd_pd(m_pri0,m_ctf,m_dat0);
        m_dat1 = _mm256_fmadd_pd(m_pri1,m_ctf,m_dat1);
        m_dat0 = _mm256_mul_pd(m_dat0,m_dat0);
        // m4 = _mm256_mul_pd(m2,m2);
        // m3 = _mm256_add_pd(m3,m4);
        m_dat0 = _mm256_fmadd_pd(m_dat1,m_dat1,m_dat0);
        //m5 = _mm256_mul_pd(m4,m_sigRcp);
        local_my_result = _mm256_fmadd_pd(m_dat0,m_sigRcp,local_my_result);
    
    }

    _mm256_storeu_pd(my_result_array,local_my_result);

    //_mm256_zeroupper();




    
    my_result=(my_result_array[0]+my_result_array[1])+(my_result_array[2]+my_result_array[3]);
    
    
    #pragma omp atomic
    result+=my_result;

 }    
    return result;


}




