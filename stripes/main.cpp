/**
   @file main.cpp 



 * Application to detect white stripes (pedestrian, humps etc.) in road scenes with C++ & OpenCV
 * @mainpage
 *
 *
 * @version: 1.1
 * 
*/

# include <stdio.h>
# include <math.h>
# include <vector>
# include "opencv2/core/core.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/imgproc/imgproc.hpp"
# include <iostream>

using namespace cv;
using namespace std;


//forward declarations
int readWriteVideo(char** argv); /// < method 1
int readWriteVideo2(char** argv); /// < method 2

/// main : invokes all methods here
int main(int argc, char** argv){

    //checks the arguments to invoke method
    if( argc < 2 )
    {
        cout << "Usage: ./detectStripes <videofile>" << endl;
        cout << "Or \nUsage: ./detectStripes <videofile> <method no>" << endl;
        return 1;
    } 

     if( argc == 3 )
    {
        cout << "Invoking method 2..." << endl;
        readWriteVideo2(argv);
        return 1;
    } 

    //default method invoked
    cout << "Invoking default method ..." << endl;
    readWriteVideo(argv);
	return 0;
}
