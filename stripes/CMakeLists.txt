cmake_minimum_required(VERSION 2.8)
project( Stripes )
find_package( OpenCV REQUIRED )
add_executable( Stripes detectStripes.cpp detectStripeBars.cpp main.cpp) #Stripes 
target_link_libraries( Stripes ${OpenCV_LIBS} ) #Stripes [same name as above]
