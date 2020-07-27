/* 
 * logDataVSPrior is a function to calculate 
 * the accumulation from ABS of two groups of complex data
 * *************************************************************************/

// based on version1，using 2D array
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
const int local_num = m/count;

inline double logDataVSPrior(const double* dat_0, const double* dat_1,const double* pri_0, const double* pri_1, const double* ctf, const double* sigRcp, const int num, const double disturb0);





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


    // double* dat_0 = (double*) memalign(32,m*8);
    // double* dat_1 = (double*) memalign(32,m*8);
    // double* pri_0 = (double*) memalign(32,m*8);
    // double* pri_1 = (double*) memalign(32,m*8);



    double** dat = new double*[2];
    for(int i=0;i<2;i++)
        dat[i] = new double[m];
    //释放空间

    double** pri = new double*[2];
    for(int i=0;i<2;i++)
        pri[i] = new double[m];
    
    double* ctf = (double*) memalign(32,m*8);
    double* sigRcp = (double*) memalign(32,m*8);




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
        // dat[i] = Complex (dat0, dat1);
        // pri[i] = Complex (pri0, pri1);
        dat[0][i]=dat0;
        dat[1][i]=dat1;
        pri[0][i]=-pri0;
        pri[1][i]=-pri1;
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
        result = logDataVSPrior(dat[0], dat[1],pri[0], pri[1],ctf, sigRcp, m, disturb[t]);
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


    for(int i=0;i<2;i++)
        delete []dat[i];
    delete []dat;

     for(int i=0;i<2;i++)
        delete []pri[i];
    delete []pri;
    free((void* )ctf);
    free((void* )sigRcp);

    delete[] disturb;
    return EXIT_SUCCESS;
}

inline double logDataVSPrior(const double* dat_0, const double* dat_1,const double* pri_0, const double* pri_1, const double* ctf, const double* sigRcp, const int num, const double disturb0)
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



    start = local_num*my_rank;
    end = start+local_num;
    
    
    __m256d m_dat0, m_dat1, m_pri0, m_pri1,m_ctf,m_sigRcp; 
    __m256d m_disturb0 = _mm256_set_pd(disturb0,disturb0,disturb0,disturb0);

    __m256d local_my_result = _mm256_set_pd(0.0,0.0,0.0,0.0);
    for (i = start; i < end; i+=4)
    {   

        m_dat0 = _mm256_load_pd(dat_0+i);
        m_dat1 = _mm256_load_pd(dat_1+i);
        m_pri0 = _mm256_load_pd(pri_0+i);
        m_pri1 = _mm256_load_pd(pri_1+i);
        m_ctf = _mm256_load_pd(ctf+i);
        m_sigRcp = _mm256_load_pd(sigRcp+i);

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




