#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define COL 32
#define ROW 64

using namespace std;

ostream& operator<<(ostream& cout, const vector<vector<int> >& vec){
    for (size_t i = 0; i < ROW; i++)
    {
        for (size_t j = 0; j < COL; j++)
        {
            cout << vec[i][j] << ' ';
        }
        cout << endl;
    }
    cout << "---------------------------------\n";
    return cout;
}

__global__
void addMul(int* m, int* n, int* l);

int main(){

    srand(int(time(0)));

    vector<vector<int> > m(ROW);

    for (size_t i = 0; i < m.size(); i++)
    {
        m[i].resize(COL);
    }

    for (size_t i = 0; i < ROW; i++)
    {
        for (size_t j = 0; j < COL; j++)
        {
            m[i][j] = rand()%10;
        }
    }

    vector<vector<int> > n(ROW);

    for (size_t i = 0; i < n.size(); i++)
    {
        n[i].resize(COL);
    }

    for (size_t i = 0; i < ROW; i++)
    {
        for (size_t j = 0; j < COL; j++)
        {
            n[i][j] = rand()%10;
        }
    }

    cout << n;
    cout << m;

    int** d_m;
    int** d_n;

    cudaMalloc((void**)&d_m, COL*ROW*sizeof(int));
    cudaMalloc((void**)&d_n, COL*ROW*sizeof(int));

    // cudaMemcpy(d_m, m.data(), COL*ROW*sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_n, n.data(), COL*ROW*sizeof(int), cudaMemcpyHostToDevice);

    // dim3 grid = make_uint3(1, 4, 1);
    // dim3 block = make_uint3(32, 16, 1);

}

__global__
void addMul(int* m, int* n, int* l){

}
