/**
@brief file OpenGLImageWarper.h
warp image using 2D mesh grid
@author Shane Yuan
@date Mar 14, 2018
*/

#include "OpenGLImageWarper.h"

OpenGLImageWarper::OpenGLImageWarper() {}
OpenGLImageWarper::~OpenGLImageWarper() {}

/**
@brief init function
init OpenGL for image warping
@return int
*/
int OpenGLImageWarper::init() {
	// Initialise GLFW
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_SAMPLES, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	GLFWwindow* window = glfwCreateWindow(1000, 1000, "hide window", NULL, NULL);
	if (window == nullptr) {
		fprintf(stderr, "Failed to open GLFW window.\n");
		glfwTerminate();
		return -1;
	}
	// hide window
	glfwHideWindow(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}
	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
	// RGBA texture blending
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// load shader
	programID = glshader::LoadShaders("E:\\Project\\OpenGLImageWarper\\shader\\TransformVertexShader.vertexshader.glsl", 
		"E:\\Project\\OpenGLImageWarper\\shader\\SolidColor.fragmentshader.glsl");



	return 0;
}

/**
@brief warp image
@param cv::Mat input: input image
@param cv::Mat & output: output image
@param cv::Size size: output size
@param cv::Mat mesh: input mesh used for warp
@return int
*/
int OpenGLImageWarper::warp(cv::Mat input, cv::Mat & output,
	cv::Size size, cv::Mat mesh) {
	// generate texture

	// generate frame buffer 
	glGenFramebuffers(1, &frameBufferID);
	return 0;
}

