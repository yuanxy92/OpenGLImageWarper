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
		@param std::string vertexShaderName: vertex shader name
		@param std::string fragShaderName: fragment shader name
		@return int
		*/
		int init(std::string vertexShaderName, std::string fragShaderName);

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
	};
};


#endif