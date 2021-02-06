#include <iostream>
#include <string>
#include <vector>
#include <math.h>

using namespace std;

// string findsame(string& s){
//     s = s + ',';
//     vector<string> myvec;
//     for(int i = 0; i < s.length(); i++){
//         std::string mys = "";
//         while(s[i] != ','){
//             mys = mys + s[i];
//             i++;
//         }
//         myvec.push_back(mys);
//         mys.clear();
//     }
    
//     string ss = myvec[0];

//     vector<int> count;
//     int tempcount = 0;

//     for (int i = 0; i < myvec.size(); i++)
//     {
//         string tems = myvec[i];
//         for (int j = 0; j < ss.length(); j++)
//         {
//             if(ss[j] != tems[j]&& tems.length() < ss.length()){
//                 break;
//             }
//             tempcount++;
            
//         }
//         count.push_back(tempcount);
//         tempcount = 0;
//     }

//     long min = pow(2,31) - 1;

//     for (int i = 0; i < count.size(); i++)
//     {
//         if (count[i] < min)
//         {
//             min = count[i];
//         }
//     }

//     string result = "";
//     for (int i = 0; i < min; i++)
//     {
//         result += ss[i];
//     }
        
//     return result;
// }


void sort(std::vector<int>& nums, int left, int right)
{
    if(right > left)
        return;
    
    int point = nums[left];    
    int L = left;
    int R = right;

    while (right > left)
    {
        while(right > left && nums[right] >= point){
            right--;
        }
        if(right > left){
            nums[left] = nums[right];
        }
        while(right > left && nums[left] <= point){
            left++;
        }
        if(right > left){
            nums[right] = nums[left];
        }
        if(right <= left){
            nums[left] = point;
        }
    }

    sort(nums, L, left);
    sort(nums, right+1, R);

}

int main()
{
    string test = "sss,sss";
    string s = findsame(test);
    cout << s ;
}