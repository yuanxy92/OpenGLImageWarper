#version 450 core

// Interpolated values from the vertex shaders
in vec2 UV;

// Ouput data
out vec4 colorRGBA;

// Values that stay constant for the whole mesh.
uniform sampler2D myTextureSampler;

void main(){
	colorRGBA = texture( myTextureSampler, UV ).rgba;
	if (colorRGBA.a < 0.95)
		discard;
	else colorRGBA.a = 1.0f;
}