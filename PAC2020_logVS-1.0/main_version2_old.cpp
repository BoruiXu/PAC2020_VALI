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
#include <immintrin.h> 
using namespace std;

// typedef complex<double> Complex;
typedef chrono::high_resolution_clock Clock;

const int m=1638400;    // DO NOT CHANGE!!
const int K=100000; // DO NOT CHANGE!!
const int local_num = m/25;

inline double logDataVSPrior(const double* dat_0, const double* dat_1,const double* pri_0, const double* pri_1, const double* ctf, const double* sigRcp, const int num, const double disturb0);
inline double logDataVSPrior2(const double* dat_0, const double* dat_1,const double* pri_0, const double* pri_1, const double* ctf, const double* sigRcp, const int num, const double disturb0);

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
        result = logDataVSPrior(dat_0, dat_1,pri_0, pri_1,ctf, sigRcp, m, disturb[t]);
        fout << t+1 << ": " << result <<"\n";
    }
    fout<<flush;
    fout.close();

    auto endTime = Clock::now(); 

    auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
    cout << "Computing time=" << compTime.count() << " microseconds" << endl;

    delete[] dat_0;
    delete[] pri_0;

    delete[] dat_1;
    delete[] pri_1;

    delete[] ctf;
    delete[] sigRcp;
    delete[] disturb;
    return EXIT_SUCCESS;
}

inline double logDataVSPrior(const double* dat_0, const double* dat_1,const double* pri_0, const double* pri_1, const double* ctf, const double* sigRcp, const int num, const double disturb0)
{
    double result = 0.0;
    #pragma omp parallel num_threads(25)
  
{

       
    int my_rank = omp_get_thread_num();
    //int thread_count = omp_get_num_threads();
    int i;
    int start,end; 
    double my_result=0.0;
    //double *local_my_result = new double[local_num];
    double local_my_result[local_num];

    start = local_num*my_rank;
    end = start+local_num;
    
    
    __m256d m_dat0, m_dat1, m_pri0, m_pri1,m_ctf,m_sigRcp,m1,m2,m3,m4,m5;  

    for (i = start; i < end; i+=4)
    {   

        m_dat0 = _mm256_loadu_pd(dat_0+i);
        m_dat1 = _mm256_loadu_pd(dat_1+i);
        m_pri0 = _mm256_loadu_pd(pri_0+i);
        m_pri1 = _mm256_loadu_pd(pri_1+i);
        m_ctf = _mm256_loadu_pd(ctf+i);
        m_sigRcp = _mm256_loadu_pd(sigRcp+i);

            //my_result+= ( norm( dat[i] - ctf[i] * pri[i] ) * sigRcp[i] );
        m1 = _mm256_mul_pd(m_pri0,m_ctf);
        m2 = _mm256_mul_pd(m_pri1,m_ctf);
        m1 = _mm256_sub_pd(m_dat0,m1);
        m2 = _mm256_sub_pd(m_dat1,m2);
        m3 = _mm256_mul_pd(m1,m1);
        m4 = _mm256_mul_pd(m2,m2);
        m3 = _mm256_add_pd(m3,m4);
        m5 = _mm256_mul_pd(m3,m_sigRcp);
        _mm256_storeu_pd(local_my_result+(i%local_num),m5);
   
    }




    for(i=0;i<local_num;i++){
        my_result+=local_my_result[i];
    }
    
    #pragma omp atomic
    result+=my_result;

 }    
    return result*disturb0;


}




// inline double logDataVSPrior2(const double* dat_0, const double* dat_1,const double* pri_0, const double* pri_1, const double* ctf, const double* sigRcp, const int num, const double disturb0)
// {

//     double result=0.0,temp1,temp2,temp3,temp4;
//     int i;

//     for(i=0;i<num;i++){
//         temp1 = (pri_0[i]*ctf[i]);
//         temp2 = (pri_1[i]*ctf[i]);

//         temp1 = (dat_0[i]-temp1);
//         temp2 = dat_1[i]-temp2;

//         temp3 = (temp1*temp1+temp2*temp2);

//         result+=(temp3*sigRcp[i]);

//     }

//     return result*disturb0;

// }
