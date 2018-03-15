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

/**
@brief error callback function
@param int error: error id
@param const char* description: error description
@return int
*/
void OpenGLImageWarper::error_callback(int error, const char* description) {
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
		"E:\\Project\\OpenGLImageWarper\\shader\\TextureFragmentShader.fragmentshader.glsl");
	// init camera
	cameraPtr = std::make_shared<GLCamParam>();
	// generate vertex arrays
	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);
	// generate vertex/uv buffer
	glGenBuffers(1, &vertexID);
	glGenBuffers(1, &uvID);
	// generate frame buffer
	glGenFramebuffers(1, &frameBufferID);
	// generate textures
	glGenTextures(1, &inputTextureID);
	glGenTextures(1, &outputTextureID);
	return 0;
}

/**
@brief release function
release OpenGL buffers
@return int
*/
int OpenGLImageWarper::release() {
	glDeleteVertexArrays(1, &vertexArrayID);
	glDeleteBuffers(1, &vertexID);
	glDeleteBuffers(1, &uvID);
	glDeleteFramebuffers(1, &frameBufferID);
	glDeleteTextures(1, &inputTextureID);
	glDeleteTextures(1, &outputTextureID);
	return 0;
}

/**
@brief generate vertex and uv buffer from mesh grid
@param cv::Mat mesh: input mesh
@param GLfloat* vertexBuffer: output vertex buffer data
@param GLfloat* uvBuffer: output buffer data
@param cv::Size textureSize: input texture size
@return int
*/
int OpenGLImageWarper::genVertexUVBufferData(cv::Mat mesh, GLfloat* vertexBuffer,
	GLfloat* uvBuffer, cv::Size textureSize) {
	// calculate size
	size_t num = 0;
	for (size_t row = 0; row < mesh.rows - 1; row++) {
		for (size_t col = 0; col < mesh.cols - 1; col++) {
			// calculate quad
			cv::Point2f tl, tr, bl, br;
			cv::Point2f tluv, truv, bluv, bruv;
			tl = mesh.at<cv::Point2f>(row, col);
			tr = mesh.at<cv::Point2f>(row, col + 1);
			bl = mesh.at<cv::Point2f>(row + 1, col);
			br = mesh.at<cv::Point2f>(row + 1, col + 1);
			tluv = cv::Point2f(tl.x / textureSize.width, tl.y / textureSize.height);
			truv = cv::Point2f(tr.x / textureSize.width, tr.y / textureSize.height);
			bluv = cv::Point2f(bl.x / textureSize.width, bl.y / textureSize.height);
			bruv = cv::Point2f(br.x / textureSize.width, br.y / textureSize.height);
			// assign data to buffer
			vertexBuffer[18 * num + 0] = tl.x; vertexBuffer[18 * num + 1] = tl.y; vertexBuffer[18 * num + 2] = 0;
			vertexBuffer[18 * num + 3] = tr.x; vertexBuffer[18 * num + 4] = tr.y; vertexBuffer[18 * num + 5] = 0;
			vertexBuffer[18 * num + 6] = br.x; vertexBuffer[18 * num + 7] = br.y; vertexBuffer[18 * num + 8] = 0;
			vertexBuffer[18 * num + 9] = br.x; vertexBuffer[18 * num + 10] = br.y; vertexBuffer[18 * num + 11] = 0;
			vertexBuffer[18 * num + 12] = bl.x; vertexBuffer[18 * num + 13] = bl.y; vertexBuffer[18 * num + 14] = 0;
			vertexBuffer[18 * num + 15] = tl.x; vertexBuffer[18 * num + 16] = tl.y; vertexBuffer[18 * num + 17] = 0;

			//uvBuffer[12 * num + 0] = tluv.x; uvBuffer[12 * num + 1] = tluv.y;
			//uvBuffer[12 * num + 2] = truv.x; uvBuffer[12 * num + 3] = truv.y;
			//uvBuffer[12 * num + 4] = bruv.x; uvBuffer[12 * num + 5] = bruv.y;
			//uvBuffer[12 * num + 6] = bruv.x; uvBuffer[12 * num + 7] = bruv.y;
			//uvBuffer[12 * num + 8] = bluv.x; uvBuffer[12 * num + 9] = bluv.y;
			//uvBuffer[12 * num + 10] = tluv.x; uvBuffer[12 * num + 11] = tluv.y;

			uvBuffer[12 * num + 0] = static_cast<float>(col) / static_cast<float>(mesh.cols - 1);
			uvBuffer[12 * num + 1] = static_cast<float>(row) / static_cast<float>(mesh.rows - 1);
			uvBuffer[12 * num + 2] = static_cast<float>(col + 1) / static_cast<float>(mesh.cols - 1);
			uvBuffer[12 * num + 3] = static_cast<float>(row) / static_cast<float>(mesh.rows - 1);
			uvBuffer[12 * num + 4] = static_cast<float>(col + 1) / static_cast<float>(mesh.cols - 1);
			uvBuffer[12 * num + 5] = static_cast<float>(row + 1) / static_cast<float>(mesh.rows - 1);
			uvBuffer[12 * num + 6] = static_cast<float>(col + 1) / static_cast<float>(mesh.cols - 1);
			uvBuffer[12 * num + 7] = static_cast<float>(row + 1) / static_cast<float>(mesh.rows - 1);
			uvBuffer[12 * num + 8] = static_cast<float>(col) / static_cast<float>(mesh.cols - 1);
			uvBuffer[12 * num + 9] = static_cast<float>(row + 1) / static_cast<float>(mesh.rows - 1);
			uvBuffer[12 * num + 10] = static_cast<float>(col) / static_cast<float>(mesh.cols - 1);
			uvBuffer[12 * num + 11] = static_cast<float>(row) / static_cast<float>(mesh.rows - 1);

			num++;
		}
	}
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
	glBindTexture(GL_TEXTURE_2D, inputTextureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, input.cols, input.rows, 
		0, GL_BGR, GL_UNSIGNED_BYTE, input.data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	bindToCudaSurface(inputTextureID, inputCudaTextureSurfaceObj);

	// generate frame buffer 
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
	// generate output texture
	glBindTexture(GL_TEXTURE_2D, outputTextureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size.width, size.height,
		0, GL_BGR, GL_UNSIGNED_BYTE, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	// bind output texture to frame buffer
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outputTextureID, 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	}
	bindToCudaSurface(outputTextureID, outputCudaTextureSurfaceObj);

	size_t vertexBufferSize, uvBufferSize;
	GLfloat* vertexBuffer;
	GLfloat* uvBuffer;
	cv::Size meshSize = mesh.size();
	size_t triangleNum = meshSize.area() * 2;
	// generate buffer data
	vertexBuffer = new GLfloat[triangleNum * 3 * 3];
	uvBuffer = new GLfloat[triangleNum * 3 * 2];
	vertexBufferSize = triangleNum * 3 * 3 * sizeof(float);
	uvBufferSize = triangleNum * 3 * 2 * sizeof(float);
	this->genVertexUVBufferData(mesh, vertexBuffer, uvBuffer, inputSize);
	glBindBuffer(GL_ARRAY_BUFFER, vertexID);
	glBufferData(GL_ARRAY_BUFFER, vertexBufferSize, vertexBuffer, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, uvID);
	glBufferData(GL_ARRAY_BUFFER, uvBufferSize, uvBuffer, GL_STATIC_DRAW);

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
	glDrawArrays(GL_TRIANGLES, 0, triangleNum * 3); // 2*3 indices starting at 0 -> 2 triangles
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	glfwSwapBuffers(window);
	glfwPollEvents();

	delete[] uvBuffer;
	delete[] vertexBuffer;

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
int OpenGLImageWarper::warp8U(cv::Mat input, cv::Mat & output,
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
	// BGR to gray
	cv::Mat inputBGR;
	cv::cvtColor(input, inputBGR, cv::COLOR_GRAY2BGR);

	// generate input texture and upload input data into OpenGL texture
	glBindTexture(GL_TEXTURE_2D, inputTextureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, inputBGR.cols, inputBGR.rows,
		0, GL_BGR, GL_UNSIGNED_BYTE, inputBGR.data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	bindToCudaSurface(inputTextureID, inputCudaTextureSurfaceObj);

	// generate frame buffer 
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
	// generate output texture
	glBindTexture(GL_TEXTURE_2D, outputTextureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size.width, size.height,
		0, GL_BGR, GL_UNSIGNED_BYTE, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	// bind output texture to frame buffer
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outputTextureID, 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	}
	bindToCudaSurface(outputTextureID, outputCudaTextureSurfaceObj);

	size_t vertexBufferSize, uvBufferSize;
	GLfloat* vertexBuffer;
	GLfloat* uvBuffer;
	cv::Size meshSize = mesh.size();
	size_t triangleNum = meshSize.area() * 2;
	// generate buffer data
	vertexBuffer = new GLfloat[triangleNum * 3 * 3];
	uvBuffer = new GLfloat[triangleNum * 3 * 2];
	vertexBufferSize = triangleNum * 3 * 3 * sizeof(float);
	uvBufferSize = triangleNum * 3 * 2 * sizeof(float);
	this->genVertexUVBufferData(mesh, vertexBuffer, uvBuffer, inputSize);
	glBindBuffer(GL_ARRAY_BUFFER, vertexID);
	glBufferData(GL_ARRAY_BUFFER, vertexBufferSize, vertexBuffer, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, uvID);
	glBufferData(GL_ARRAY_BUFFER, uvBufferSize, uvBuffer, GL_STATIC_DRAW);

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
	glDrawArrays(GL_TRIANGLES, 0, triangleNum * 3); // 2*3 indices starting at 0 -> 2 triangles
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	glfwSwapBuffers(window);
	glfwPollEvents();

	delete[] uvBuffer;
	delete[] vertexBuffer;

	return 0;
}