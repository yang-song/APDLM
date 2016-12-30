/* Code written by Yang Song (yangsong@cs.stanford.edu)
 * Dynamic programming for loss augmented inference of Average Precision
 * DP<true>: positive update
 * DP<false>: negative update
*/
#include "dp.h"
#include <cassert>
template<bool positive=false>
double DP(const vector<double> &P, const vector<double> &N, vector<size_t> &pos, const double epsilon){
    /*
     * The scores need to be sorted in descending order before
     * calling this function!
    */
    //std::sort(P.begin(),P.end(),[](double a,double b){return (a > b) ? true:false;});
    //std::sort(N.begin(),N.end(),[](double a,double b){return (a > b) ? true:false;});

    int m = P.size();
    int n = N.size();
    assert(m > 0);
    assert(n > 0);
    vector<vector<double>> f(m+1,vector<double>(n+1,0));
    vector<vector<double>> d(m+1,vector<double>(n+1,0));
    vector<vector<double>> F(m+1,vector<double>(n+1,0));
    vector<vector<double>> G(m+1,vector<double>(n+1,0));
    pos.resize(m + n);
    fill(pos.begin(),pos.end(),0);

    for(int i = 1;i <= m;i++)
        for(int j = 1;j <= n;j++){
            F[i][j] = F[i][j-1] - 1.0/m/n * (P[i-1] - N[j-1]);
            G[i][j] = G[i-1][j] + 1.0/m/n * (P[i-1] - N[j-1]);
        }

    const int flag = positive ? -1 : 1;
    for(int i = 0;i <= m;i++)
        for(int j = 0;j <= n;j++){
            if(i == 0 && j == 0){
                f[i][j] = 0;
                d[i][j] = 0;
                continue;
            }
            if(i == 1 && j == 0){
                f[i][j] = 1.0/m*epsilon * flag;
                d[i][j] = 1;
                continue;
            }
            if(i == 0 && j == 1){
                f[i][j] = 0;
                d[i][j] = -1;
                continue;
            }
            if(i == 0){
                f[i][j] = f[i][j-1] + G[i][j];
                d[i][j] = -1;
                continue;
            }
            if(j == 0){
                f[i][j] = f[i-1][j] + epsilon* 1.0/m * i / double(i+j) * flag + F[i][j];
                d[i][j] = 1;
                continue;
            }
            if(f[i][j-1] + G[i][j] > f[i-1][j] + epsilon * 1.0/m * i / double(i+j) * flag + F[i][j]){
                f[i][j] = f[i][j-1] + G[i][j];
                d[i][j] = -1;
            }
            else{
                f[i][j] = f[i-1][j] + epsilon * 1.0/m * i / double(i+j) * flag + F[i][j];
                d[i][j] = 1;
            }
        }

    //recover the ranking
    for(int i = m,j = n; i >= 0 && j >= 0 && !(i == 0 && j == 0);){
        if(d[i][j] == 1){
            pos[i - 1] = i + j - 1;
            i--;
        }
        else{
            pos[j + m - 1] = i + j - 1;
            j--;
        }
    }

    return f[m][n] - epsilon*flag;
}

template double DP<true>(const vector<double>&, const vector<double>&, vector<size_t>&, const double);
template double DP<false>(const vector<double>&, const vector<double>&,vector<size_t>&, const double);

