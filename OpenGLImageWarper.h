/**
@brief file OpenGLImageWarper.h
warp image using 2D mesh grid
@author Shane Yuan
@date Mar 14, 2018
*/

#ifndef __OPENGL_IMAGE_WARPER__
#define __OPENGL_IMAGE_WARPER__

// include stl
#include <memory>
#include <cstdlib>
// include GLEW
#include <GL/glew.h>
// include GLFW
#include <glfw3.h>
// include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
// include cuda
#ifdef _WIN32
#include <windows.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <surface_functions.h>
// include shader and texture loader
#include <common/shader.hpp>

// include opencv
#include <opencv2/opencv.hpp>

class OpenGLImageWarper {
private:
	// program ID
	GLuint programID;

	// frame buffer ID
	GLuint frameBufferID;
public:

private:

public:
	OpenGLImageWarper();
	~OpenGLImageWarper();

	/**
	@brief init function
		init OpenGL for image warping
	@return int
	*/
	int init();

	/**
	@brief warp image
	@param cv::Mat input: input image
	@param cv::Mat & output: output image
	@param cv::Size size: output size
	@param cv::Mat mesh: input mesh used for warp
	@return int
	*/
	int warp(cv::Mat input, cv::Mat & output, 
		cv::Size size, cv::Mat mesh);
};


#endif