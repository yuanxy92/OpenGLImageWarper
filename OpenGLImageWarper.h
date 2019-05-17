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
#include <GLFW/glfw3.h>
// include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// include shader and texture loader
#include <common/shader.hpp>

// include opencv
#include <opencv2/opencv.hpp>

namespace gl {
	// opengl camera class
	class GLCamParam {
	public:
		float x;
		float y;
		float z;
		float camera_near;
		float camera_far;
		float fov;
		float aspect;
		glm::mat4 projection;

		GLCamParam() : x(0), y(0), z(0), camera_near(0.1), camera_far(10.0 * 1500.0),
			fov(90), aspect(1.0) {
			projection = glm::perspective(glm::radians(fov), aspect,
				camera_near, camera_far);
		}
		GLCamParam(float x, float y, float z, float aspect) {
			this->x = x; this->y = y; this->z = z;
			this->camera_near = 0.1; this->camera_far = 10.0 * 1500.0;
			this->fov = 90; this->aspect = aspect;
			projection = glm::perspective(glm::radians(fov), aspect,
				camera_near, camera_far);
		}
		int reCalcProj() {
			projection = glm::perspective(glm::radians(fov), aspect,
				camera_near, camera_far);
			return 0;
		}
		int project(float inputx, float inputy, float & outx, float & outy) {
			float winWidth = z * 2 * aspect;
			float winHeight = z * 2;
			outx = (inputx - x) / winWidth + 0.5;
			outy = (inputy - y) / winHeight + 0.5;
			//printf("position :(%f, %f, %f) win size: (%f, %f)\n", x, y, z, winWidth, winHeight);
			//printf("input :(%f, %f) output: (%f, %f)\n", inputx, inputy, outx, outy);
			return 0;
		}
		~GLCamParam() {}
	};

	// opengl image warper class
	class OpenGLImageWarper {
	private:
		// window
		GLFWwindow* window;
		// camera
		std::shared_ptr<GLCamParam> cameraPtr;
		// program ID
		GLuint programID;
		// frame buffer ID
		GLuint frameBufferID;
		// texture ID and size
		GLuint inputTextureID;
		cv::Size inputSize;
		GLuint outputTextureID;
		cv::Size outputSize;
		// vertex array ID
		GLuint vertexArrayID;
		// vertex ID
		GLuint vertexID;
		// UV ID
		GLuint uvID;

	public:
		static const std::string _defalutVertexShader;
		static const std::string _defalutFragmentShader;

		enum class ShaderLoadMode
		{
			FilePath = 0,
			Content = 1
		};
	private:
		/**
		@brief error callback function
		@param int error: error id
		@param const char* description: error description
		@return int
		*/
		static void error_callback(int error, const char* description);

		/**
		@brief generate vertex and uv buffer from mesh grid
		@param cv::Mat mesh: input mesh
		@param GLfloat* vertexBuffer: output vertex buffer data
		@param GLfloat* uvBuffer: output buffer data
		@param cv::Size textureSize: input texture size
		@return int
		*/
		int genVertexUVBufferData(cv::Mat mesh, GLfloat* vertexBuffer,
			GLfloat* uvBuffer, cv::Size textureSize);

		/**
		@brief generate vertex and uv buffer from mesh grid
		@param cv::Mat mesh: input mesh
		@param GLfloat* vertexBuffer: output vertex buffer data
		@param GLfloat* uvBuffer: output buffer data
		@param cv::Size textureSize: input texture size
		@return int
		*/
		int genVertexUVBufferDataBack(cv::Mat mesh, GLfloat* vertexBuffer,
			GLfloat* uvBuffer, cv::Size textureSize);

	public:
		OpenGLImageWarper();
		~OpenGLImageWarper();

		/**
		@brief init function
			init OpenGL for image warping
		@param std::string vertexShaderName: vertex shader 
		@param std::string fragShaderName: fragment shader 
		@return int
		*/
		int init(std::string vertexShaderName, std::string fragShaderName, ShaderLoadMode mode = ShaderLoadMode::FilePath);

		/**
		@brief release function
			release OpenGL buffers
		@return int
		*/
		int release();

		/**
		@brief warp image
		@param cv::Mat input: input image
		@param cv::Mat & output: output image
		@param cv::Size size: output size
		@param cv::Mat mesh: input mesh used for warp
		@param int direction: direction of warping
			0: warp forward, 1: warp backward
		@return int
		*/
		int warp(cv::Mat input, cv::Mat & output,
			cv::Size size, cv::Mat mesh, int direction = 0);

		template<typename T>
		int warpSingle(cv::Mat input, cv::Mat & output,
			cv::Size size, cv::Mat mesh, int direction = 0, int interpolation = GL_LINEAR);//GL_NEAREST

		/**
		@brief debug function
		@return int
		*/
		int debug();

		/**
		@brief transfer normal mesh to real size mesh
		@param cv::Mat mesh: input CV_64FC2/CV_32FC2 mesh with normalized coordinates
		@param int width: input real image width
		@param int height: input real image height
		@return cv::Mat: return CV_32FC2 mesh with real image size
		*/
		static cv::Mat meshNoraml2Real(cv::Mat mesh, int width, int height);

		template<typename T>
		static T getImgSubpix(const cv::Mat& img, cv::Point2f pt, int borderType = cv::BORDER_CONSTANT, int interpolation = cv::INTER_LINEAR);

		template<typename T>
		static T getImgSubPixNormalized(const cv::Mat& img, cv::Point2f pt, int borderType = cv::BORDER_CONSTANT, int interpolation = cv::INTER_LINEAR);
	};

	template<typename T>
	inline int OpenGLImageWarper::warpSingle(cv::Mat input, cv::Mat & output, cv::Size size, cv::Mat mesh, int direction, int interpolation)
	{
		if (!(std::is_same<unsigned char, T>::value || std::is_same<unsigned short, T>::value || std::is_same<float, T>::value || std::is_same<unsigned int, T>::value))
		{
			std::cout <<
				"Only uchar, ushort, uint or float is supported in single mode" <<
				std::endl;
			return -1;
		}
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
		unsigned int dataType;

		if (std::is_same<unsigned char, T>::value)
		{
			dataType = GL_UNSIGNED_BYTE;
		}
		else if (std::is_same<unsigned short, T>::value)
		{
			dataType = GL_UNSIGNED_SHORT;
		}
		else if (std::is_same<unsigned int, T>::value)
		{
			dataType = GL_UNSIGNED_INT;
		}
		else
		{
			dataType = GL_FLOAT;
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, input.cols, input.rows,
			0, GL_RED, dataType, input.data);
		
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, interpolation);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, interpolation);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

		// generate frame buffer 
		glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
		// generate output texture
		glBindTexture(GL_TEXTURE_2D, outputTextureID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size.width, size.height,
			0, GL_BGR, GL_UNSIGNED_BYTE, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, interpolation);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, interpolation);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		// bind output texture to frame buffer
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outputTextureID, 0);
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
		}

		size_t vertexBufferSize, uvBufferSize;
		GLfloat* vertexBuffer;
		GLfloat* uvBuffer;
		cv::Size meshSize = cv::Size(mesh.size().width - 1, mesh.size().height - 1);
		size_t triangleNum = meshSize.area() * 2;
		// generate buffer data
		vertexBuffer = new GLfloat[triangleNum * 3 * 3];
		uvBuffer = new GLfloat[triangleNum * 3 * 2];
		vertexBufferSize = triangleNum * 3 * 3 * sizeof(float);
		uvBufferSize = triangleNum * 3 * 2 * sizeof(float);
		if (direction == 0) {
			this->genVertexUVBufferData(mesh, vertexBuffer, uvBuffer, inputSize);
		}
		else if (direction == 1) {
			this->genVertexUVBufferDataBack(mesh, vertexBuffer, uvBuffer, inputSize);
		}
		else {
			std::cout << "ERROR::Input parameter:: only 0 and 1 are support for direction." << std::endl;
			exit(-1);
		}

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

		T* pixels = new T[size.width * size.height];
		glReadPixels(0, 0, size.width, size.height, GL_RED, dataType, pixels);
		cv::Mat img;// (size, CV_8UC4, pixels);
		if (std::is_same<unsigned char, T>::value)
		{
			cv::Mat tmp(size, CV_8U, pixels);
			img = tmp.clone();
		}
		else if (std::is_same<unsigned short, T>::value)
		{
			cv::Mat tmp(size, CV_16U, pixels);
			img = tmp.clone();
		}
		else if (std::is_same<unsigned int, T>::value)
		{
			cv::Mat tmp(size, CV_32S, pixels);
			img = tmp.clone();
		}
		else
		{
			cv::Mat tmp(size, CV_32F, pixels);
			img = tmp.clone();
		}
		//cv::cvtColor(img, output, cv::COLOR_RGBA2BGR);
		output = img.clone();
		delete[] uvBuffer;
		delete[] vertexBuffer;
		delete[] pixels;

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		return 0;
	}

	template<typename T>
	inline T OpenGLImageWarper::getImgSubpix(const cv::Mat & img, cv::Point2f pt, int borderType, int interpolation)
	{
		cv::Mat patch;
		cv::remap(img, patch, cv::Mat(1, 1, CV_32FC2, &pt), cv::noArray(),
			interpolation, borderType, cv::Scalar(0));
		return patch.at<T>(0, 0);
	}

	template<typename T>
	inline T OpenGLImageWarper::getImgSubPixNormalized(const cv::Mat & img, cv::Point2f pt, int borderType , int interpolation)
	{
		return getImgSubpix<T>(img, cv::Point2f(pt.x * img.cols, pt.y *img.rows), borderType, interpolation);
	}
};


#endif