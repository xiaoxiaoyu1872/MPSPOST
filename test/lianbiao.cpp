#include <stdio.h>

struct ListNode
{
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(NULL){}
};

class Solution
{
private:
    /* data */
public:
    ListNode* reverseList(ListNode* head)
    {
        ListNode* tmp(0);
        tmp = head;
        while (head)
        {
            head = head->next;
            head->next = tmp;
            tmp = head->next; 
        }
        

        return tmp;
    }
    Solution(/* args */);
    ~Solution();
};

Solution::Solution(/* args */)
{
}

Solution::~Solution()
{
}


int main()
{
    ListNode a(1);
    ListNode b(2);
    ListNode c(3);
    ListNode d(4);

    Solution s;

    a.next = &b;
    b.next = &c;
    c.next = &d; 
    d.next = NULL;
    
    ListNode* head = &a;
    
    while (head)
    {
        printf("%d\n", head->val);
        head = head->next;
    }
    
    d.next = &c;
    c.next = &b;
    b.next = &a;
    a.next = NULL;
    head = &d;

    while (head)
    {
        printf("%d\n", head->val);
        head = head->next;
    }

    ListNode* rehead;
    rehead = s.reverseList(head);

        while (rehead)
    {
        printf("%d\n", rehead->val);
        rehead = rehead->next;
    }
}


