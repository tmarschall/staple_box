#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

int main()
{
  double a,b;
  srand(time(0));

  clock_t start = clock();
  for (int i = 0; i < 100; i++)
    {  
      a = (double)rand()/(double)RAND_MAX;
      b = -a;

      cout << a << " "  << b << endl;

      double c = 100.*(a - b)-100.;
      c += 100.0 * ((c <= -50.0) - (c >= 50.0));
      //if (c >= 50.0)
      //	c -= 100.0;
      //else if (c <= -50.0)
      //  c += 100.0;
      
      cout << a << " " << b << " " << c << " " << double(c > 0) << endl;
    }
  clock_t stop = clock();

  cout << "\nRun time: " << stop - start << endl;

  return 0;
}
