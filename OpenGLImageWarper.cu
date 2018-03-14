/**
@brief file OpenGLImageWarper.cu
warp image using 2D mesh grid
@author Shane Yuan
@date Mar 14, 2018
*/

#include "OpenGLImageWarper.h"

using namespace gl;

/**
@brief copy surface texture to OpenCV GPU mat
@param cudaSurfaceObject_t texture: input cuda surface object binded to rendered texture
@param cv::cuda::PtrStep<uchar3> img_d: output OpenCV GPU mat
@param int width: texture width
@param int height: texture height
@return int
*/
__global__ void OpenGLImageWarperKernel::copy_surface_to_gpumat(cudaSurfaceObject_t texture,
	cv::cuda::PtrStep<uchar3> img_d, int width, int height) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		uchar4 input;
		surf2Dread(&input, texture, x * 4, height - y - 1);
		img_d.ptr(y)[x] = make_uchar3(input.z, input.y, input.x);
	}
}

/**
@brief download rendered textures to gpu mat
@return int
*/
int OpenGLImageWarper::debug() {
	cv::cuda::GpuMat input_d(inputSize, CV_8UC3);
	cv::cuda::GpuMat output_d(outputSize, CV_8UC3);
	cv::Mat input_h, output_h;

	// input 
	dim3 dimBlock(32, 32);
	dim3 dimGrid((inputSize.width + dimBlock.x - 1) / dimBlock.x,
		(inputSize.height + dimBlock.y - 1) / dimBlock.y);
	OpenGLImageWarperKernel::copy_surface_to_gpumat << <dimGrid, dimBlock >> >(
		inputCudaTextureSurfaceObj, input_d, inputSize.width, inputSize.height);

	// output
	dim3 dimBlock2(32, 32);
	dim3 dimGrid2((outputSize.width + dimBlock.x - 1) / dimBlock.x,
		(outputSize.height + dimBlock.y - 1) / dimBlock.y);
	OpenGLImageWarperKernel::copy_surface_to_gpumat << <dimGrid2, dimBlock2 >> >(
		outputCudaTextureSurfaceObj, output_d, outputSize.width, outputSize.height);

	input_d.download(input_h);
	output_d.download(output_h);

	return 0;
}