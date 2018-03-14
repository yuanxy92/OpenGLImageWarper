#version 450 core

// Interpolated values from the vertex shaders
in vec2 UV;

// Ouput data
out vec4 colorRGBA;

// Values that stay constant for the whole mesh.
uniform sampler2D myTextureSampler;

void main(){
	//if (UV.x > 0 && UV.x < 1 && UV.y > 0 && UV.y < 1)
	//	colorRGBA = vec4(1.0, 0.0, 0.0, 1.0);
	// Output color = color of the texture at the specified UV
	colorRGBA = texture( myTextureSampler, UV ).rgba;
	if (colorRGBA.a < 0.95)
		discard;
	else colorRGBA.a = 1.0f;
}