#include <iostream>

int main(void)
{
#pragma omp parallel for default(none) shared(std::cout)
    for (size_t i = 0; i < 1024; ++i)
	{
	    std::cout << "i: " << i << std::endl;
	}

    return 0;
}
