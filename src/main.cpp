/*************************************************************************
	> File Name: main.cpp
	> Author: 
	> Mail: 
	> Created Time: Thu 22 Feb 2024 02:54:47 PM CST
 ************************************************************************/

#include<iostream>
#include "hello.h"
using namespace std;

int main(int argc, char const *argv[])
{
	hello::sayHelloto(argv[1]);
	return 0;
}
