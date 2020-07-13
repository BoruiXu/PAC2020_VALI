/* 
 * logDataVSPrior is a function to calculate 
 * the accumulation from ABS of two groups of complex data
 * *************************************************************************/

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <omp.h>
#include <intrin.h> 
using namespace std;

// typedef complex<double> Complex;
typedef chrono::high_resolution_clock Clock;

const int m=1638400;    // DO NOT CHANGE!!
const int K=100000; // DO NOT CHANGE!!

inline double logDataVSPrior(const double* dat_0, const double* dat_1,const double* pri_0, const double* pri_1, const double* ctf, const double* sigRcp, const int num, const double disturb0);

int main ( int argc, char *argv[] )
{ 
    // Complex *dat = new Complex[m];
    // Complex *pri = new Complex[m];
    double *dat_0 = new double[m];
    double *dat_1 = new double[m];
    double *pri_0 = new double[m];
    double *pri_1 = new double[m];
    double *ctf = new double[m];
    double *sigRcp = new double[m];
    double *disturb = new double[K];
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
        result = logDataVSPrior(dat, pri, ctf, sigRcp, m, disturb[t]);
        fout << t+1 << ": " << result <<"\n";
    }
    fout<<flush;
    fout.close();

    auto endTime = Clock::now(); 

    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Computing time=" << compTime.count() << " microseconds" << endl;

    delete[] dat;
    delete[] pri;

    delete[] ctf;
    delete[] sigRcp;
    delete[] disturb;
    return EXIT_SUCCESS;
}

inline double logDataVSPrior(const double* dat_0, const double* dat_1,const double* pri_0, const double* pri_1, const double* ctf, const double* sigRcp, const int num, const double disturb0)
{
    double result = 0.0;
    double temp;
    int local_num = 65536;
    #pragma omp parallel num_threads(25)
    {
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    int i,start,end，loop = local_num/8;
    double my_result=0.0;
    double local_dat_0[local_num],local_dat_1[local_num],local_pri_0[local_num],local_pri_1[local_num],local_ctf[local_num],local_sigRcp[local_num],local_my_result[local_num];

    __m521d m1, m2, m3, m4,m5,m6,m7;  
    start = my_rank*local_num;
    end = start+local_num;

    //************************************
    copy(dat_0+start,dat_0+end,local_dat_0);
    cpoy(dat_1+start,dat_1+end,local_dat_1);
    copy(pri_0+start,pri_0+end,local_pri_0);
    copy(pri_1+start,pri_1+end,local_pri_1);
    copy(ctf+start,ctf+end,local_ctf);
    copy(sigRcp+start,sigRcp+end,local_sigRcp);
    //************************************

    __m512d* p_local_dat_0 = (__m512d*)local_dat_0;
    __m512d* p_local_dat_1 = (__m512d*)local_dat_1;
    __m512d* p_local_pri_0 = (__m512d*)local_pri_0;
    __m512d* p_local_pri_1 = (__m512d*)local_pri_1;
    __m512d* p_local_ctf = (__m512d*)local_ctf;
    __m512d* p_local_sigRcp = (__m512d*)local_sigRcp;
    __m512d* p_local_my_reslt = (__m512d*)local_my_result;


    for (i = 0; i < loop; i++)
    {

            //my_result+= ( norm( dat[i] - ctf[i] * pri[i] ) * sigRcp[i] );
        m1 = _mm512_mul_pd(*p_local_pri_0,*p_local_ctf);
        m2 = _mm512_mul_pd(*p_local_pri_1,*p_local_ctf);
        m3 = _mm512_sub_pd(*p_local_dat_0,m1);
        m4 = _mm512_sub_pd(*p_local_dat_1,m2);
        m5 = _mm512_mul_pd(m3,m3);
        m6 = _mm512_mul_pd(m4,m4);
        m7 = _mm512_add_pd(m5,m6);
        *p_local_my_result = _mm_mul512_pd(m7,*p_local_sigRcp);
   
    }

    local_my_result = (double*) p_local_my_result;


    for(i=0;i<local_num;i++){
        my_result+=local_my_result[i];
    }
    
    #pragma omp atomic
    result+=my_result;

    }   
    return result*disturb0;
}
