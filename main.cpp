
#include "OpenGLImageWarper.h"

int main(int argc, char* argv[]) {

	gl::OpenGLImageWarper warper;
	cv::Mat input = cv::imread("E:\\data\\giga\\18cameras\\00.jpg");
	cv::resize(input, input, cv::Size(1000, 1000));
	cv::Mat output;
	cv::Mat mesh;

	warper.init();
	warper.warp(input, output, input.size(), mesh);
	//warper.debug();

	return 0;
}