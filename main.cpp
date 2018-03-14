
#include "OpenGLImageWarper.h"

int main(int argc, char* argv[]) {

	gl::OpenGLImageWarper warper;
	cv::Mat input = cv::imread("E:\\data\\giga\\18cameras\\00.jpg");
	cv::resize(input, input, cv::Size(1000, 1000));
	cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);
	cv::Mat output;
	cv::Mat mesh(3, 3, CV_32FC2);

	//mesh.at<cv::Point2f>(0, 0) = cv::Point2f(-20, -20);
	//mesh.at<cv::Point2f>(0, 1) = cv::Point2f(1000, 0);
	//mesh.at<cv::Point2f>(1, 0) = cv::Point2f(20, 980);
	//mesh.at<cv::Point2f>(1, 1) = cv::Point2f(950, 910);

	mesh.at<cv::Point2f>(0, 0) = cv::Point2f(-20, -20);
	mesh.at<cv::Point2f>(0, 1) = cv::Point2f(500, 10);
	mesh.at<cv::Point2f>(0, 2) = cv::Point2f(1000, 0);
	mesh.at<cv::Point2f>(1, 0) = cv::Point2f(0, 500);
	mesh.at<cv::Point2f>(1, 1) = cv::Point2f(470, 500);
	mesh.at<cv::Point2f>(1, 2) = cv::Point2f(980, 500);
	mesh.at<cv::Point2f>(2, 0) = cv::Point2f(20, 980);
	mesh.at<cv::Point2f>(2, 1) = cv::Point2f(420, 950);
	mesh.at<cv::Point2f>(2, 2) = cv::Point2f(950, 910);

	warper.init();
	warper.warp8U(input, output, input.size(), mesh);
	cv::Mat out = warper.getWarpedImg8U();
	warper.release();

	return 0;
}