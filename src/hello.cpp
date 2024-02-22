/*************************************************************************
	> File Name: hello.cpp
	> Author: 
	> Mail: 
	> Created Time: Thu 22 Feb 2024 02:54:40 PM CST
 ************************************************************************/

#include<iostream>
#include "hello.h"
using namespace std;

void hello::sayHelloto(const std::string& name)
{
	std::cout << "Hello, " << name << "!" << std::endl;
}
