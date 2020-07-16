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
using namespace std;

typedef complex<double> Complex;
typedef chrono::high_resolution_clock Clock;

const int m=1638400;	// DO NOT CHANGE!!
const int K=100000;	// DO NOT CHANGE!!

double logDataVSPrior(const Complex* dat, const Complex* pri, const double* ctf, const double* sigRcp, const int num, const double disturb0);

int main ( int argc, char *argv[] )
{ 
    Complex *dat = new Complex[m];
    Complex *pri = new Complex[m];
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
        dat[i] = Complex (dat0, dat1);
        pri[i] = Complex (pri0, pri1);
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

inline double logDataVSPrior(const Complex* dat, const Complex* pri, const double* ctf, const double* sigRcp, const int num, const double disturb0)
{
    double result = 0.0;
    int local_num = 204800;
    #pragma omp parallel num_threads(8)
    {
	double my_result=0.0,temp2;
    double my_res0=0.0,my_res1=0.0,my_res2=0.0,my_res3=0.0;
	Complex temp1;
	int my_rank = omp_get_thread_num();
	int thread_count = omp_get_num_threads();
	int i,start,end;

	start = my_rank*local_num;
	end = start+local_num;

	#pragma omp simd
   	for (i = start; i < end; i+=4)
   	 {
	    temp1 = dat[i]-ctf[i]*pri[i];
	    temp2 = norm(temp1);
      	//my_result+= (temp2 * sigRcp[i] );
        my_res0 +=(temp2 * sigRcp[i] );

        temp1 = dat[i+1]-ctf[i+1]*pri[i+1];
        temp2 = norm(temp1);
        //my_result+= (temp2 * sigRcp[i] );
        my_res0 +=(temp2 * sigRcp[i+1] );


        temp1 = dat[i+2]-ctf[i+2]*pri[i+2];
        temp2 = norm(temp1);
        //my_result+= (temp2 * sigRcp[i] );
        my_res0 +=(temp2 * sigRcp[i+2] );

        temp1 = dat[i+3]-ctf[i+3]*pri[i+3];
        temp2 = norm(temp1);
        //my_result+= (temp2 * sigRcp[i] );
        my_res0 +=(temp2 * sigRcp[i+3] );

   
    }

    my_result=(my_res0+my_res1)+(my_res2+my_res3);
	#pragma omp critical
	result+=my_result;

    }	
    return result*disturb0;
}
