/**
@brief file OpenGLImageWarper.cpp
warp image using 2D mesh grid
@author Shane Yuan
@date Mar 14, 2018
*/

#include "OpenGLImageWarper.h"

using namespace gl;

OpenGLImageWarper::OpenGLImageWarper() {}
OpenGLImageWarper::~OpenGLImageWarper() {}

/**
@brief bind opengl texture to cuda surface
@param GLuint textureID: id of input opengl texture
@param cudaSurfaceObject_t & surfceObj: output cuda surface object
@return int
*/
int OpenGLImageWarper::bindToCudaSurface(GLuint textureID, cudaSurfaceObject_t & surfaceObj) {
	// bind texture to cuda surface 
	cudaError cudaStatus;
	cudaArray* cuImg;
	struct cudaGraphicsResource* cudaTextureRes;
	cudaStatus = cudaGraphicsGLRegisterImage(&cudaTextureRes, textureID,
		GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
	if (cudaStatus != cudaSuccess) {
		std::cout << "map opengl texture to cuda failed! " << std::endl;
		std::cout << cudaGetErrorString(cudaStatus) << std::endl;
		return -1;
	}
	cudaStatus = cudaGraphicsMapResources(1, &cudaTextureRes, 0);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaGraphicsMapResources failed!" << std::endl;
		std::cout << cudaGetErrorString(cudaStatus) << std::endl;
		exit(-1);
	}
	cudaStatus = cudaGraphicsSubResourceGetMappedArray(&cuImg, cudaTextureRes, 0, 0);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaGraphicsSubResourceGetMappedArray failed!" << std::endl;
		std::cout << cudaGetErrorString(cudaStatus) << std::endl;
		exit(-1);
	}
	// Specify surface
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	// Create the surface objects
	resDesc.res.array.array = cuImg;
	cudaStatus = cudaCreateSurfaceObject(&surfaceObj, &resDesc);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaCreateSurfaceObject failed!" << std::endl;
		std::cout << cudaGetErrorString(cudaStatus) << std::endl;
		exit(-1);
	}
	return 0;
}

static void error_callback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}

/**
@brief init function
init OpenGL for image warping
@return int
*/
int OpenGLImageWarper::init() {
	// Initialise GLFW
	glfwSetErrorCallback(error_callback);

	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_SAMPLES, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(1000, 1000, "hide window", NULL, NULL);
	if (window == nullptr) {
		fprintf(stderr, "Failed to open GLFW window.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	// hide window
	// glfwHideWindow(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}
	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.3f, 0.0f);
	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
	// RGBA texture blending
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// load shader
	programID = glshader::LoadShaders("E:\\Project\\OpenGLImageWarper\\shader\\TransformVertexShader.vertexshader.glsl", 
		"E:\\Project\\OpenGLImageWarper\\shader\\TextureFragmentShader.fragmentshader.glsl");

	// init camera
	cameraPtr = std::make_shared<GLCamParam>();

	// generate vextex and uv buffer 
	// The fullscreen quad's FBO
	cv::Size size(1000, 1000);
	static const GLfloat g_quad_vertex_buffer_data[] = {
		0.0f, 0.0f, 0.0f,
		size.width, 0.0f, 0.0f,
		size.width, size.height, 0.0f,
		size.width, size.height, 0.0f,
		0.0f, size.height, 0.0f,
		0.0f, 0.0f, 0.0f
	};
	// The fullscreen quad's FBO
	static const GLfloat g_quad_uv_buffer_data[] = {
		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f,  1.0f,
		1.0f,  1.0f,
		0.0f, 1.0f,
		0.0f,  0.0f
	};
	// init vertex arrays
	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);
	// bind
	glGenBuffers(1, &vertexID);
	glBindBuffer(GL_ARRAY_BUFFER, vertexID);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data),
		g_quad_vertex_buffer_data, GL_STATIC_DRAW);
	glGenBuffers(1, &uvID);
	glBindBuffer(GL_ARRAY_BUFFER, uvID);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_uv_buffer_data),
		g_quad_uv_buffer_data, GL_STATIC_DRAW);

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
	// get input and output size
	inputSize = input.size();
	outputSize = size;

	// adjust camera position
	cameraPtr->x = size.width / 2;
	cameraPtr->y = size.height / 2;
	cameraPtr->z = size.height / 2;
	cameraPtr->aspect = static_cast<float>(size.width) /
		static_cast<float>(size.height);
	cameraPtr->reCalcProj();

	// generate input texture and upload input data into OpenGL texture
	glGenTextures(1, &inputTextureID);
	glBindTexture(GL_TEXTURE_2D, inputTextureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, input.cols, input.rows, 
		0, GL_BGR, GL_UNSIGNED_BYTE, input.data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	this->bindToCudaSurface(inputTextureID, inputCudaTextureSurfaceObj);

	// generate frame buffer 
	glGenFramebuffers(1, &frameBufferID);
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
	// generate output texture
	glGenTextures(1, &outputTextureID);
	glBindTexture(GL_TEXTURE_2D, outputTextureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size.width, size.height,
		0, GL_BGR, GL_UNSIGNED_BYTE, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	// bind output texture to frame buffer
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outputTextureID, 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	}
	this->bindToCudaSurface(outputTextureID, outputCudaTextureSurfaceObj);

	//this->debug();

	// draw mesh
	// Render to the fbo
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
	glViewport(0, 0, size.width, size.height);
	// Clear the screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// Use our shader
	GLuint matrixInShader = glGetUniformLocation(programID, "MVP");
	GLuint textureInShader = glGetUniformLocation(programID, "myTextureSampler");
	// in the "MVP" uniform
	glm::mat4 View = glm::lookAt(
		glm::vec3(cameraPtr->x, cameraPtr->y, cameraPtr->z), // camera position 
		glm::vec3(cameraPtr->x, cameraPtr->y, 0), // look at position 
		glm::vec3(0, 1, 0)  // head is up (set to 0,-1,0 to look upside-down)
	);
	glm::mat4 Model = glm::mat4(1.0);
	glm::mat4 MVP = cameraPtr->projection * View * Model;


	glUseProgram(programID);
	glUniformMatrix4fv(matrixInShader, 1, GL_FALSE, &MVP[0][0]);
	// bind texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->inputTextureID);
	glUniform1i(textureInShader, 0);
	// 1st attribute buffer : vertexs 
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexID);
	glVertexAttribPointer(
		0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
		);
	// 2nd attribute buffer : UVs
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, uvID);
	glVertexAttribPointer(
		1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
		2,                                // size : U+V => 2
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
		);
	// Draw the triangles
	glDrawArrays(GL_TRIANGLES, 0, 2 * 3); // 2*3 indices starting at 0 -> 2 triangles
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	glfwSwapBuffers(window);
	glfwPollEvents();

	this->debug();

	//system("pause");
	return 0;
}

