// // // #include <iostream>
// // // #include <stdio.h>
// // // #include <vector>

// // // // // class Solution
// // // // // {
// // // // // private:
// // // // //     /* data */
// // // // // public:
// // // // //     Solution(/* args */);
// // // // //     ~Solution();
// // // // //     std::vector< std::vector<int> > subset(std::vector<int>& nums){
// // // // //         std::vector< std::vector<int> > result;
// // // // //         return result;
// // // // //     }
// // // // // };

// // // // // Solution::Solution(/* args */)
// // // // // {
// // // // // }

// // // // // Solution::~Solution()
// // // // // {
// // // // // }


// // // // // int main(){
// // // // //     Solution s;

// // // // //     std::vector<int> nums;
// // // // //     nums.push_back(1);
// // // // //     nums.push_back(2);
// // // // //     nums.push_back(3);
// // // // //     std::vector< std::vector<int> > result;
// // // // //     result = s.subset(nums);
// // // // // }

// // // // #include<iostream>
// // // // #include<vector>

// // // // void merage(std::vector<int> &vec1, std::vector<int> &vec2, std::vector<int> &vec){    
// // // //     int i = 0;
// // // //     int j = 0;
// // // //     while(i < vec1.size() && j < vec2.size()){
// // // //         if(vec1[i] <= vec2[j]){
// // // //             vec.push_back(vec1[i]);
// // // //             i++;
// // // //         }
// // // //         else{
// // // //             vec.push_back(vec2[j]);
// // // //             j++;
// // // //         }
// // // //     }
// // // //     for(;i < vec1.size(); i++){
// // // //         vec.push_back(vec1[i]);
// // // //     }
// // // //     for(;j < vec2.size(); j++){
// // // //         vec.push_back(vec2[j]);
// // // //     }
// // // // } 

// // // // void sortmerge(std::vector<int> &vec){
// // // //     if(vec.size() < 2){
// // // //         return;
// // // //     }   
// // // //     int mid = vec.size()/2;
// // // //     std::vector<int> vec1;
// // // //     std::vector<int> vec2;
// // // //     for(int i = 0; i < mid; i++)
// // // //     {
// // // //         vec1.push_back(vec[i]);
// // // //     }
// // // //     for (int i = mid; i < vec.size(); i++)
// // // //     {
// // // //         vec2.push_back(vec[i]);
// // // //     }

// // // //     sortmerge(vec1);
// // // //     sortmerge(vec2);
    
// // // //     vec.clear();
// // // //     merage(vec1, vec2, vec);

// // // // }


// // // // int main(){

// // // //     // std::vector<int> test1;
// // // //     // test1.push_back(7);
// // // //     // test1.push_back(1);
// // // //     // test1.push_back(2);
// // // //     // test1.push_back(5);
// // // //     // test1.push_back(1);
// // // //     // test1.push_back(3);



// // // //     std::vector<int> test;
// // // //     test.push_back(7);
// // // //     test.push_back(1);
// // // //     test.push_back(2);
// // // //     test.push_back(5);
// // // //     test.push_back(1);
// // // //     test.push_back(3);



// // // //     // std::vector<int> test2;



// // // //     for(auto i : test){
// // // //         std::cout << i;
// // // //     }
// // // //     std::cout << std::endl;

// // // //     // merage(test, test1 ,test2);
// // // //     sortmerge(test);


// // // //     for(auto i : test){
// // // //         std::cout << i;
// // // //     }
// // // //     std::cout << std::endl;
// // // // }


// // // struct TreeNode
// // // {
// // //     int val;
// // //     TreeNode* left;
// // //     TreeNode* right;
// // //     TreeNode(int x): val(x), left(nullptr) ,right(nullptr){}
// // // };

// // // void prinftree(TreeNode* node){
// // //     if(!node){
// // //         return;
// // //     }
// // //     std::cout<< ' ' << node->val;
// // //     prinftree(node->left);
// // //     prinftree(node->right);
// // // }


// // // int main(){
// // //     TreeNode a(10);
// // //     TreeNode b(20);
// // //     TreeNode c(30);
// // //     TreeNode d(40);
// // //     TreeNode f(50);
// // //     a.left = &b;
// // //     b.left = &c;
// // //     a.right = &d;
// // //     d.right = &f;
// // //     prinftree(&a);
// // // }


// // /**
// //  * Definition for a binary tree node.
// //  * struct TreeNode {
// //  *     int val;
// //  *     TreeNode *left;
// //  *     TreeNode *right;
// //  *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
// //  * };
// //  */
// // #include<iostream>
// // #include<vector>
// // #include<stdio.h>

// // struct TreeNode {
// //     int val;
// //     TreeNode *left;
// //     TreeNode *right;
// //     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
// // };

// // void preorder(TreeNode* root, TreeNode* target, int& finish, std::vector<TreeNode*>& path, std::vector<std::vector<TreeNode*> >& result){
// //     if(!root || finish){
// //         return;
// //     }

// //     path.push_back(root);
// //     if(root == target){
// //         result.push_back(path);
// //         finish = 1;
// //         // path.clear();
// //         return;
// //     }

// //     preorder(root->left, target, finish, path, result);
// //     preorder(root->right, target, finish, path, result);
    
// //     path.pop_back();
// // }

// // class Solution {
// // public:
// //     TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
// //         std::vector<TreeNode*> path;
// //         std::vector<std::vector<TreeNode*> > result;
// //         int finish = 0;
// //         preorder(root, p, finish, path, result);
// //         path.clear();
// //         finish = 0;
// //         preorder(root, q, finish, path, result);

// //         int max = result[0].size() > result[1].size() ? result[1].size() : result[0].size();

// //         TreeNode* head;

// //         for(int i = max - 1; i >= 0; i--)
// //         {
// //             if(result[0][i] == result[1][i]){
// //                 head = result[0][i];
// //                 return head;
// //             }
// //         }

// //         // for(int i = 0; i < max; i++){
// //         //     if(result[0][i] == result[1][i]){
// //         //         head = result[0][i];
// //         //     }
// //         // }

// //         return head;
// //     }
// // };


// // int main(){
// //     TreeNode a(3);
// //     TreeNode b(5);
// //     TreeNode c(1);
// //     TreeNode d(6);
// //     TreeNode e(2);
// //     TreeNode f(0);
// //     TreeNode x(8);
// //     TreeNode y(7);
// //     TreeNode z(4);

// //     a.left = &b;
// //     a.right = &c;
// //     b.left = &d;
// //     b.right = &e;
// //     c.left = &f;
// //     c.right = &x;
// //     e.left = &y;
// //     e.right = &z;


// //     Solution solve;
// //     TreeNode* result = solve.lowestCommonAncestor(&a, &b, &f);
// //     printf("lowestCommonAncestor = %d\n", result->val);
// // }

// // #include <iostream>
// // #include <stack>
// // #include <queue>
// // #include <queue>
// // // struct cmp(int a, int b)
// // // {
// // //     return a<b;
// // // }

// // struct tmp
// // {
// //     bool operator()(int a, int b)
// //     {
// //         return a < b;
// //     }
// // };

// // struct TreeNode
// // {
// //     int val;
// //     TreeNode* left;
// //     TreeNode* right;
// //     TreeNode(int x): val(x), left(nullptr), right(nullptr){}
// // };


// // int main(){
// //     std::priority_queue<int, std::vector<int>, std::greater<int> > small_heap;

// //     std::priority_queue<int, std::vector<int>, tmp > smal1_heap;

// //     std::queue<TreeNode*> Q;
// //     Q.push
// // }


// // #include <iostream>
// // #include <string>
// // using namespace std;

// // // void test01(){
// // //     pair<string, int> p("Tom", 20);
// // //     cout << 
// // // }

// // int main()
// // {
// //     std::cout << INT_MIN << std::endl;
// // }

// #include<iostream>
// #include<vector>
// #include<stdio.h>
// #include<map>
// #include<queue>

// #include<algorithm>
// // struct ListNode
// // {
// //     int val;
// //     ListNode* next;
// //     ListNode(int x): val(x), next(nullptr){}
// // };

// // int hash_func(int key, int table_len){
// //     return key%table_len;
// // }

// // void insert(ListNode* hash_table[], ListNode* node, int table_len){
// //     int hash_key = hash_func(node->val, table_len);

// //     node->next = hash_table[hash_key];
// //     hash_table[hash_key] = node;
// // }

// // bool search(ListNode* hash_table[], int value, int table_len){
// //     int hash_key = hash_func(value, table_len);
// //     ListNode* head = hash_table[hash_key];

// //     while(head){
// //         if(head->val == value){
// //             return true;
// //         }
// //         head = head->next;
// //     }
// //     return false;
// // }


// // using namespace std;
// // int main(){
// //     // const int TABLE_LEN = 11;
// //     // ListNode* hash_table[TABLE_LEN] = {0};
// //     // std::vector<ListNode*> hash_node_vec;
// //     // int test[8] = {1,1,4,9,20,30,150,500};
// //     // for(int i = 0; i < 8; i++){
// //     //     hash_node_vec.push_back(new ListNode(test[i]));
// //     // }
// //     // for (int i = 0; i < hash_node_vec.size(); i++){
// //     //     insert(hash_table, hash_node_vec[i], TABLE_LEN);
// //     // }

// //     // for(int i = 0; i < TABLE_LEN; i++){
// //     //     printf("[%d]:", i);
// //     // }    

// //     // for(int i = 0; i < 8; i++){
// //     //     delete hash_node_vec[i];
// //     // }
// //     std::map<std::string, int> hash_map;
// //     string s1 = "aaa";
// //     string s2 = "bbb";
// //     string s3 = "ccc";

// //     hash_map[s1] = 1;
// //     hash_map[s2] = 2;
// //     hash_map[s3] = 3;

// //     // if(hash_map.find("aaa") != hash_map.end()){
// //     //     printf("find");
// //     // }

// //     // for(int i = 0; i < hash_map.size(); i++)
// //     // {
// //     //     printf("%s %d", hash_map[i].first)
// //     // }

// //     std::map<std::string, int>::iterator it;

// //     // for(it = hash_map.begin(); it != hash_map.end(); it++){
// //     //     printf("%s : %d \n", it->first.data(), it->second);
// //     // }

// //     string s = "";
// //     if(s == "")
// //     {
// //         cout << s << "here";
// //     }
// //     // char a = 'a';
// //     // string sss("");
// //     // sss = sss + a;

// //     const char* a = "aaaaa";
// //     string ssss = a;

// //     char b[] = "bbbbbb";
// //     string bbbb = b;

// //     std::sort(bbbb.begin(), bbbb.end());

// //     int&& youzhi = 12+5;
// //     int fas= youzhi;
// //     std::cout << fas << endl;
// //     youzhi = 1;
// //     std::cout << youzhi << endl;
// //     std::priority_queue <int, std::vector<int>, std::greater<int>> mysmall;
// //     mysmall.push(youzhi);

// //     mysmall.top();
// //     mysmall.pop();
// // }

// #include<vector>
// #include<stdio.h>
// #include<cstring>

// class MyString
// {
// private:
//     int _len;
//     char* _data;
//     void init_data(const char* str)
//     {
//         _data = new char[_len+1];
//         memcpy(_data,str,_len);
//         _data[_len] = '\0';
//     }
// public:
//     MyString()
//     {
//         _len = 0;
//         _data = NULL;
//         std::cout << "MyString()\n";
//     }
//     MyString(const char* str)
//     {
//         _len = strlen(str);
//         init_data(str);
//         std::cout << "const char* str\n";
//     }

//     MyString(const MyString& rhs)
//     {
//         _len = rhs._len;
//         init_data(rhs._data);
//     }

//     MyString& operator=(MyString& rhs)
//     {
//         if(this != &rhs)
//         {
//             _len = rhs._len;
//             init_data(rhs._data);
//             //MyString(rhs);
//         }
//         return *this;

//         std::cout << "MyString& rhs\n";
//     }
//     virtual ~MyString()
//     {
//         if(_data) free(_data);
//     }

//     MyString(MyString&& rhs)
//     {
//         _len = rhs._len;
//         _data = rhs._data;
//         rhs._len = 0;
//         rhs._data = NULL;
//         std::cout << "MyString&& rhs\n";
//     }

//     MyString& operator=(MyString&& rhs)
//     {
//         if(this != &rhs)
//         {
//             _len = rhs._len;
//             _data = rhs._data;
//             rhs._len = 0;
//             rhs._data = NULL;
//         }
//         return *this;
//         std::cout << "operator=(MyString&& rhs)\n ";
//     }



// };


// int main()
// {
//     MyString a;
//     a = MyString("Hello");
//     std::vector<MyString> vec;
//     vec.push_back(MyString("World"));

//     std::vector<int>* myvec = new std::vector<int>();
//     myvec->push_back(1);
//     myvec->push_back(1);
//     myvec->push_back(1);
//     myvec->push_back(1);
//     myvec->push_back(1);
//     myvec->push_back(1);
// }


// #include <stdio.h>
// #include <iostream>
// #include <math.h>
// #include <cuda_runtime.h>
// #include <vector>


// __global__ void reduce(int* d_myvec, int* d_myred)
// {
//     int threadId = threadIdx.x + blockDim.x * blockIdx.x;
//     int blockId = blockIdx.x;
// }


// __global__ void reduce0(int *g_idata, int *g_odata) {
// extern __shared__ int sdata[];
//     // each thread loads one element from global to shared mem
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
//     sdata[tid] = g_idata[i];
//     __syncthreads();


//     // do reduction in shared mem
//     for(unsigned int s=1; s < blockDim.x; s *= 2) {
//         if (tid % (2*s) == 0) 
//         {
//             sdata[tid] += sdata[tid + s];
//         }
//     __syncthreads();

//     }
//     // write result for this block to global mem
//     if (tid == 0) 
//     g_odata[blockIdx.x] = sdata[0];
// } 




// int main()
// {
//     std::vector<int> myvec;
//     for (size_t i = 0; i < 1000000; i++)
//     {
//         myvec.push_back(i);
//     }
    
//     int* d_myvec;
//     cudaMalloc((void**)d_myvec, sizeof(myvec));
//     cudaMemcpy((void**)d_myvec, myvec.data(), sizeof(myvec), cudaMemcpyHostToDevice);

//     int numthread = 256;
//     dim3 block, grid;
//     block = make_uint3(numthread, 1, 1);
//     int blocknum = ceil(1000000/numthread);
//     grid = make_uint3(blocknum, 1, 1);
    
//     int* d_myred;
//     cudaMalloc((void**)d_myred, blocknum*sizeof(int));

//     reduce<<<grid, block>>> (d_myvec, d_myred);
// }


// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

template<class T>
void swap(T& a, T& b)
{
    if(a > b){
        T temp;
        temp = b;
        b = a;
        a = temp;
    }
}


template<class T>
bool swap_charu(T& a, T& b)
{
    if(a > b){
        T temp;
        temp = b;
        b = a;
        a = temp;
        return true;
    }
    else{
        return false;
    }
}


void sort_maopao(std::vector<int>& nums)
{
    for(int i = 0; i < nums.size() - 1; i++)
    {
        for(int j = 0; j < nums.size() - i - 1; j++)
        {
            swap(nums[j], nums[j+1]);
        }
    }
}


void sort_charu(std::vector<int>& nums)
{
    for(int i = 1; i < nums.size() - 1; i++)
    {
        bool flag = true;
        int j = i;
        while(flag && (j - 1)>=0 )
        {
            flag = swap_charu(nums[j - 1], nums[j]);
            j--;
        }
    }
}


// void kuaisu(std::vector<int>& nums, int left, int right)
// {
//     if(left >= right){
//         return;
//     }

//     int backleft = left;
//     int backright = right;

//     int index = right;
//     int point = nums[left];

//     while(left < right){
//         // while(left < right && nums[right] >= point){
//         //     right--;
//         // }
//         // if(left < right && nums[right] < point){
//         //     nums[index] = nums[right];
//         //     index = right;
//         //     right--;
//         // }
//         // while(left < right && nums[left] <= point){
//         //     left++;
//         // }
//         // if(left < right && nums[left] > point){
//         //     nums[index] = nums[left];
//         //     index = left;
//         //     left++;
//         // }
//         while(left < right && nums[right] >= point){
//             right--;
//         }
//         if(left < right){
//             nums[left] = nums[right];
//         }
//         while(left < right && nums[left] <= point){
//             left++;
//         }
//         if(left < right){
//             nums[right] = nums[left];
//         }

//         if(left >= right){
//             nums[left] = point;
//         }

//     }


//     kuaisu(nums, backleft, left - 1);
//     kuaisu(nums, right + 1, backright);
// }















void kuaisu(std::vector<int>& nums, int left, int right)
{
    if(left >= right){
        return;
    }

    int L = left;
    int R = right;
    int point = nums[left];

    while(right > left){
        while(right > left && nums[right] >= point)
        {
            right--;
        }
        if(right > left){
            nums[left] = nums[right];
        }
        while(right > left && nums[left] <= point ){
            left++;
        }
        if(right > left){
            nums[right] = nums[left];
        }

        if(left >= right){
            nums[left] = point;
        }
    }

    kuaisu(nums, L, left);
    kuaisu(nums, right + 1, R);
}





















// void merage(std::vector<int>& nums1, std::vector<int>& nums2, std::vector<int>& nums)
// {
//     // std::vector<int> nums;
//     int i = 0;
//     int j = 0;
//     while(i < nums1.size() && j < nums2.size()){
//         if(nums1[i] > nums2[j]){
//             nums.push_back(nums2[j]);
//             j++;
//         }else{
//             nums.push_back(nums1[i]);
//             i++;
//         }
//     }

//     for(;i < nums1.size(); i++){
//         nums.push_back(nums1[i]);
//     }

//     for(;j < nums2.size(); j++){
//         nums.push_back(nums2[j]);
//     }

// }


// // void merage_sort(std::vector<int>& nums)
// // {
// //     if(nums.size() < 2)
// //         return;

// //     int mid = nums.size()/2;

// //     std::vector<int> nums1;
// //     std::vector<int> nums2;
// //     for(int i = 0; i < mid; i++){
// //         nums1.push_back(nums[i]);
// //     }
// //     for(int i = mid; i < nums.size(); i++){
// //         nums2.push_back(nums[i]);
// //     }

// //     merage_sort(nums1);
// //     merage_sort(nums2);

// //     nums.clear();
// //     merage(nums1, nums2, nums);
// // }



// void merage_sort(std::vector<int>& nums)
// {   
//     if(nums.size() < 2){
//         return;
//     }

//     int mid = nums.size()/2;
//     std::vector<int> nums1;
//     std::vector<int> nums2;

//     for(int i = 0; i < mid; i++){
//         nums1.push_back(nums[i]);
//     }
//     for(int i = mid; i < nums.size(); i++){
//         nums2.push_back(nums[i]);
//     }

//     merage_sort(nums1);
//     merage_sort(nums2);

//     nums.clear();

//     merage(nums1, nums2, nums);
// }






void printf(const std::vector<int>& nums)
{
    for(auto i : nums){
        std::cout << ' '<< i;
    }
    std::cout << std::endl;
}


__global__ 
void matMul(int* a, int* b, int* c, int n)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if(threadId >= n*n)
        return;

    int col = threadId % n;
    int row = threadId / n;

    for (int i = 0; i < n; i++)
    {
        c[threadId] += a[n * row + i] * b[col + (n - 1 - i)*n];
    }
}


__global__
void reduce(int* in, int* out, int n)
{
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int mytid = threadIdx.x + blockIdx.x * blockDim.x;

    if(mytid >= n)
        return;

    sdata[tid] = in[mytid];
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0;s >>= 1){
        if(tid < s){
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0)
    out[blockIdx.x] = sdata[0];
}


__global__
void scan(int* in, int* out, int n)
{
    int mytid = threadIdx.x + blockDim.x * blockIdx.x;
    if(mytid >= n)
        return;
    
    for (int i = 1; i <= 4; i = i * 2)
    {
        if((mytid - i) >= 0){
            in[mytid] = in[mytid] + in[mytid - i];
        }
        __syncthreads();
    }

}


int main()
{
    // srand(int(time(0)));

    // std::vector<int> nums;
    // int i = 0;
    // while(i < 30){
    //     nums.push_back(rand()%30);
    //     i++;
    // }

    // printf(nums);

    // // sort_charu(nums);

    // kuaisu(nums, 0, nums.size() - 1);

    // // merage_sort(nums);

    // printf(nums);
    // int n = 2;

    // srand(int(time(0)));

    // std::vector<int> a;
    // std::vector<int> b;

    // for(int i = 0; i < n; i++){
    //     for (int j = 0; j < n; j++)
    //     {
    //         a.push_back(rand()%5);
    //     }
    // }

    // for(int i = 0; i < n; i++){
    //     for (int j = 0; j < n; j++)
    //     {
    //         b.push_back(rand()%5);
    //     }
    // }
    
    // int* d_a;
    // int* d_b;
    // int* d_c;

    // cudaMalloc((void**)&d_a, sizeof(int) * n*n);
    // cudaMalloc((void**)&d_b, sizeof(int) * n*n);
    // cudaMalloc((void**)&d_c, sizeof(int) * n*n);

    // cudaMemset(d_c, 0, sizeof(int)*n*n);

    // cudaMemcpy(d_a, a.data(), sizeof(int) * n*n, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, b.data(), sizeof(int) * n*n, cudaMemcpyHostToDevice);

    // int numthreads = 1;
    // dim3 block = make_uint3(numthreads, 1, 1);

    // int numblocks = ceil(float(n*n/numthreads));
    // dim3 grid = make_uint3(numblocks, 1, 1);

    // matMul<<<grid, block>>>(d_a, d_b, d_c, n);

    // std::vector<int> c;
    // c.resize(n*n);
    // cudaMemcpy(c.data(), d_c, sizeof(int) * n*n, cudaMemcpyDeviceToHost);

    // printf(a);
    // printf(b);
    // printf(c);

    // cudaFree(d_a);
    // cudaFree(d_b);
    // cudaFree(d_c);

    // std::vector<int> d()

    int number = 100;
    int n = number;
    int threadnum = 16;
    int blocknum = number % threadnum == 0 ?  number/threadnum : number/threadnum + 1;

    srand(int(time(0)));

    std::vector<int> a;
    for(int i = 0; i < 100; i++)
    {
        a.push_back(1);
    }

    int* d_in;
    int* d_out;

    int size = number*sizeof(int);
    int size_out = blocknum*sizeof(int);

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size_out);

    cudaMemcpy(d_in, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, size_out);

    int sharedmem = size_out;

    dim3 block = make_uint3(threadnum, 1, 1);
    dim3 grid = make_uint3(blocknum, 1, 1);

    // reduce<<<grid, block, sharedmem>>>(d_in, d_out, n);
    scan <<<grid, block, sharedmem>>>(d_in, d_out, n);

    std::vector<int> out;
    out.resize(n);

    cudaMemcpy(out.data(), d_in, size, cudaMemcpyDeviceToHost);

    printf(a);
    printf(out);

    // cudaMemcpy(out.data(), d_out, size_out, cudaMemcpyDeviceToHost);

    // printf(a);
    // printf(out);

    // cudaFree(d_in);
    // cudaFree(d_out);
}

