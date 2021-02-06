#include <iostream>
#include <vector>
#include <queue>

struct TreeNode
{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};


void BFS_print(TreeNode* root){
    std::queue<TreeNode *> Q;
    Q.push(root);
    while(!Q.empty()){
        TreeNode *node = Q.front();
        Q.pop();

        printf("");
        
        if(node->left){
            Q.push(node->left);
        }
        if(node->right){
            Q.push(node->right);
        }
    }
}






