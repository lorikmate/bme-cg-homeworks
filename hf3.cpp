//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const int tessellationLevel = 20;
float gtstart = 0.0f;
float gtend = 0.0f;
bool isVirusAlive = true;

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 90.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 40;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2)*asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp*bp / (bp - fp), 0);
	}

	void Animate(float t) { }
};

//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;
};

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos;

	void Animate(float t) {	}
};

//---------------------------
class CheckerBoardTexture : public Texture {
	//---------------------------
public:
	CheckerBoardTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 0.8, 0, 1), blue(0, 0.2, 0, 0.2);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 0) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
class SimpleRedTexture : public Texture {
	//---------------------------
public:
	SimpleRedTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 red(1.0f, 0.0f, 0.2f, 1); 
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = red;
		}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
class worldSphereTexture : public Texture {
	//---------------------------
public:
	worldSphereTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 red(0.08f, 0.5f, 0.74f, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = red;
		}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
struct RenderState {
	//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material *         material;
	std::vector<Light> lights;
	Texture *          texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

//---------------------------
class GouraudShader : public Shader {
	//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
		
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform mat4  MVP, M, Minv;  // MVP, Model, Model-inverse
		uniform Light[8] lights;     // light source direction 
		uniform int   nLights;		 // number of light sources
		uniform vec3  wEye;          // pos of eye
		uniform Material  material;  // diffuse, specular, ambient ref

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space

		out vec3 radiance;		    // reflected radiance

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;	
			vec3 V = normalize(wEye * wPos.w - wPos.xyz);
			vec3 N = normalize((Minv * vec4(vtxNorm, 0)).xyz);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein

			radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += material.ka * lights[i].La + (material.kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		in  vec3 radiance;      // interpolated radiance
		out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	GouraudShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
class PhongShader : public Shader {
	//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
class NPRShader : public Shader {
	//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform	vec4  wLightPos;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal, wView, wLight;				// in world space
		out vec2 texcoord;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		uniform sampler2D diffuseTexture;

		in  vec3 wNormal, wView, wLight;	// interpolated
		in  vec2 texcoord;
		out vec4 fragmentColor;    			// output goes to frame buffer

		void main() {
		   vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
		   float y = (dot(N, L) > 0.5) ? 1 : 0.5;
		   if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
		   else						 fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
		}
	)";
public:
	NPRShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniform(state.lights[0].wLightPos, "wLightPos");
	}
};

//---------------------------
struct VertexData {
	//---------------------------
	vec3 position, normal;
	vec2 texcoord;
	vec3 rU, rV;
};

//---------------------------
class Geometry {
	//---------------------------
protected:
	unsigned int vao, vbo;        // vertex array object
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	virtual void Redraw() {}
	virtual std::vector<VertexData> *getChildrenVtxData() {
		return NULL;
	}
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
	unsigned int nVtxPerStrip, nStrips;
public:
	std::vector<VertexData> vtxData;	// vertices on the CPU
	std::vector<VertexData> childrenVtxData;

	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual VertexData GenVertexData(float u, float v) = 0;

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, vtxData.size() * sizeof(VertexData), &vtxData[0], GL_STREAM_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void BuildTractri() {
		float phi = 0.0f;
		float stepN = 0.1f;
		float step = M_PI / 10;
		int number = 15.0f;
		for (int i = 0; i <= nStrips; i++) {
			float angle = (float)i * step;
			int n = round(number * sinf(angle));
			int M = n;
			for (int a = 0; a < M; a++) {
				childrenVtxData.push_back(GenVertexData((float)a / M, phi));
			}
			phi += stepN;
		}
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
struct Clifford {
	//---------------------------
	float f, d;
	Clifford(float f0 = 0, float d0 = 0) { f = f0, d = d0; }
	Clifford operator+(Clifford r) { return Clifford(f + r.f, d + r.d); }
	Clifford operator-(Clifford r) { return Clifford(f - r.f, d - r.d); }
	Clifford operator*(Clifford r) { return Clifford(f * r.f, f * r.d + d * r.f); }
	Clifford operator/(Clifford r) {
		float l = r.f * r.f;
		return (*this) * Clifford(r.f / l, -r.d / l);
	}
};

Clifford T(float t) { return Clifford(t, 1); }
Clifford Sin(Clifford g) { return Clifford(sin(g.f), cos(g.f) * g.d); }
Clifford Cos(Clifford g) { return Clifford(cos(g.f), -sin(g.f) * g.d); }
Clifford Tan(Clifford g) { return Sin(g) / Cos(g); }
Clifford Log(Clifford g) { return Clifford(logf(g.f), 1 / g.f * g.d); }
Clifford Exp(Clifford g) { return Clifford(expf(g.f), expf(g.f) * g.d); }
Clifford Pow(Clifford g, float n) { return Clifford(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }
Clifford Cosh(Clifford g) { return Clifford(cosh(g.f), sinh(g.f) * g.d); }
Clifford Sinh(Clifford g) { return Clifford(sinh(g.f), cosh(g.f) * g.d); }
Clifford Tanh(Clifford g) { return Clifford(Sinh(g) / Cosh(g)); }

//---------------------------
class Sphere : public ParamSurface {
	//---------------------------
	float tStart, tEnd, R = 1.0f;
public:
	Sphere() { create(); BuildTractri(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
	
		Clifford U(u*(float)2.0f*M_PI, 1), V(v*(float)M_PI, 0);
		Clifford X = Cos(U) * Sin(V);
		Clifford Y = Sin(U) * Sin(V);
		Clifford Z = Cos(V);
		vd.position = vec3(R*X.f, R*Y.f, R*Z.f);

		vec3 drdU = vec3(X.d, Y.d, Z.d);
		vd.rU = normalize(drdU);
		U.d = 0, V.d = 1;
		X = Cos(U) * Sin(V);
		Y = Sin(U) * Sin(V);
		Z = Cos(V);
		vec3 drdV = vec3(X.d, Y.d, Z.d);
		vd.rV = normalize(drdV);

		vd.normal = cross(drdU, drdV);
		vd.texcoord = vec2(u, v);
		return vd;
	}

	void Redraw() {
		tStart = gtstart;
		tEnd = gtend;
		vtxData.clear();
		create();
	}

	std::vector<VertexData> *getChildrenVtxData() {
		return &childrenVtxData;
	}

};

class NonParamSurface : public Geometry {
	unsigned int nVtxPerStrip, nStrips;
public:
	std::vector<VertexData> vtxData;
	std::vector<VertexData> halfPoints;
	NonParamSurface() { nVtxPerStrip = nStrips = 0; }

	void create() {
		glBufferData(GL_ARRAY_BUFFER, vtxData.size() * sizeof(VertexData), &vtxData[0], GL_STREAM_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		//glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, vtxData.size());
	}
};

int hvkLevel = 2;

class Tetrahedron : public NonParamSurface {
	float tStart, tEnd;
public:
	Tetrahedron() {
		Build();
	}

	void Build() {
		VertexData vd1, vd2, vd3, vd4, vd5, vd6, vd7, vd8, vd9, vd10, vd11, vd12;
		//Az origóban lévõ reguláris tetraéder 4 pontjának koordinátájának forrása: https://en.wikipedia.org/wiki/Tetrahedron
		vec3 v1 = vec3(1.0f, 0.0f, -0.7071f);
		vec3 v2 = vec3(-1.0f, 0.0f, -0.7071f);
		vec3 v3 = vec3(0.0f, 1.0f, 0.7071f);
		vec3 v4 = vec3(0.0f, -1.0f, 0.7071f);
		vd1.position = v1;
		vd2.position = v2;
		vd3.position = v3;
		vd4.position = v2;
		vd5.position = v3;
		vd6.position = v4;
		vd7.position = v3;
		vd8.position = v4;
		vd9.position = v1;
		vd10.position = v4;
		vd11.position = v1;
		vd12.position = v2;

		vec3 face1Normal = normalize(cross(vd1.position - vd2.position, vd2.position - vd3.position));
		vec3 face2Normal = normalize(cross(vd4.position - vd5.position, vd6.position - vd5.position)); 
		vec3 face3Normal = normalize(cross(vd7.position - vd8.position, vd8.position - vd9.position));
		vec3 face4Normal = normalize(cross(vd11.position - vd12.position, vd10.position - vd12.position)); 

		vd1.normal = vd2.normal = vd3.normal = face1Normal; 
		vd4.normal = vd5.normal = vd6.normal = face2Normal; 
		vd7.normal = vd8.normal = vd9.normal = face3Normal; 
		vd10.normal = vd11.normal = vd12.normal = face4Normal; 

		vtxData.push_back(vd1); 
		vtxData.push_back(vd2); 
		vtxData.push_back(vd3); 
		
		vtxData.push_back(vd4); 
		vtxData.push_back(vd5); 
		vtxData.push_back(vd6); 
		
		vtxData.push_back(vd7); 
		vtxData.push_back(vd8); 
		vtxData.push_back(vd9); 
		
		vtxData.push_back(vd10); 
		vtxData.push_back(vd11);
		vtxData.push_back(vd12);

		if (hvkLevel >= 1) {
			float _height = 0.5f + 1.5f*fabsf(sinf(2 * tEnd));
			float hgt = 0.5f * _height;
			float _hgt = 0.7f * _height;

			std::vector<VertexData> smaller;
			
			std::vector<VertexData> first = createSpike(vd1, vd2, vd3, face1Normal, _height);
			vtxData.insert(vtxData.end(), first.begin(), first.end());

			smaller = createSpike(vd1, first[0], first[4], face1Normal, hgt);
			vtxData.insert(vtxData.end(), smaller.begin(), smaller.end());

			smaller = createSpike(vd2, first[0], first[1], face1Normal, hgt);
			vtxData.insert(vtxData.end(), smaller.begin(), smaller.end());

			smaller = createSpike(vd3, first[4], first[1], face1Normal, hgt);
			vtxData.insert(vtxData.end(), smaller.begin(), smaller.end());

			std::vector<VertexData> second = createSpike(vd4, vd5, vd6, face2Normal, _height);
			vtxData.insert(vtxData.end(), second.begin(), second.end());

			smaller = createSpike(vd4, second[0], second[4], face2Normal, hgt);
			vtxData.insert(vtxData.end(), smaller.begin(), smaller.end());

			smaller = createSpike(vd5, second[0], second[1], face2Normal, hgt);
			vtxData.insert(vtxData.end(), smaller.begin(), smaller.end());

			smaller = createSpike(vd6, second[4], second[1], face2Normal, hgt);
			vtxData.insert(vtxData.end(), smaller.begin(), smaller.end());

			std::vector<VertexData> third = createSpike(vd7, vd8, vd9, face3Normal, _height);
			vtxData.insert(vtxData.end(), third.begin(), third.end());

			smaller = createSpike(vd7, third[0], third[4], face3Normal, hgt);
			vtxData.insert(vtxData.end(), smaller.begin(), smaller.end());

			smaller = createSpike(vd8, third[0], third[1], face3Normal, hgt);
			vtxData.insert(vtxData.end(), smaller.begin(), smaller.end());

			smaller = createSpike(vd9, third[4], third[1], face3Normal, hgt);
			vtxData.insert(vtxData.end(), smaller.begin(), smaller.end());

			std::vector<VertexData> fourth = createSpike(vd10, vd11, vd12, face4Normal, _height);
			vtxData.insert(vtxData.end(), fourth.begin(), fourth.end());

			smaller = createSpike(vd10, fourth[0], fourth[4], face4Normal, hgt);
			vtxData.insert(vtxData.end(), smaller.begin(), smaller.end());

			smaller = createSpike(vd11, fourth[0], fourth[1], face4Normal, hgt);
			vtxData.insert(vtxData.end(), smaller.begin(), smaller.end());

			smaller = createSpike(vd12, fourth[4], fourth[1], face4Normal, hgt);
			vtxData.insert(vtxData.end(), smaller.begin(), smaller.end());

			std::vector<VertexData> data;
			data.insert(data.end(), first.begin(), first.end());
			data.insert(data.end(), second.begin(), second.end());
			data.insert(data.end(), third.begin(), third.end());
			data.insert(data.end(), fourth.begin(), fourth.end());

			if (hvkLevel == 2) {
				std::vector<VertexData> temp;
				
				temp = createSpike(data[0], data[1], data[2], data[0].normal, _hgt);
				vtxData.insert(vtxData.end(), temp.begin(), temp.end());
				temp.clear();
				temp = createSpike(data[3], data[4], data[5], -data[3].normal, _hgt);
				vtxData.insert(vtxData.end(), temp.begin(), temp.end());
				temp.clear();
				temp = createSpike(data[6], data[7], data[8], data[6].normal, _hgt);
				vtxData.insert(vtxData.end(), temp.begin(), temp.end());
				temp.clear();
				
				temp = createSpike(data[9], data[10], data[11], -data[9].normal, _hgt);
				vtxData.insert(vtxData.end(), temp.begin(), temp.end());
				temp.clear();
				temp = createSpike(data[12], data[13], data[14], data[12].normal, _hgt);
				vtxData.insert(vtxData.end(), temp.begin(), temp.end());
				temp.clear();
				temp = createSpike(data[15], data[16], data[17], -data[15].normal, _hgt);
				vtxData.insert(vtxData.end(), temp.begin(), temp.end());
				temp.clear();
				
				temp = createSpike(data[18], data[19], data[20], data[18].normal, _hgt);
				vtxData.insert(vtxData.end(), temp.begin(), temp.end());
				temp.clear();
				temp = createSpike(data[21], data[22], data[23], -data[21].normal, _hgt);
				vtxData.insert(vtxData.end(), temp.begin(), temp.end());
				temp.clear();
				temp = createSpike(data[24], data[25], data[26], data[24].normal, _hgt);
				vtxData.insert(vtxData.end(), temp.begin(), temp.end());
				temp.clear();
				
				temp = createSpike(data[27], data[28], data[29], -data[27].normal, _hgt);
				vtxData.insert(vtxData.end(), temp.begin(), temp.end());
				temp.clear();
				temp = createSpike(data[30], data[31], data[32], data[30].normal, _hgt);
				vtxData.insert(vtxData.end(), temp.begin(), temp.end());
				temp.clear();
				temp = createSpike(data[33], data[34], data[35], -data[33].normal, _hgt);
				vtxData.insert(vtxData.end(), temp.begin(), temp.end());
				temp.clear();
			}
		}

		create();
	}

	std::vector<VertexData> createSpike(VertexData _v1, VertexData _v2, VertexData _v3, vec3 normal, float height) {
		std::vector<VertexData> result;

		VertexData vdh12, vdh23, vdh31;
		vdh12.position = vec3((_v1.position.x + _v2.position.x) / 2.0f, (_v1.position.y + _v2.position.y) / 2.0f, (_v1.position.z + _v2.position.z) / 2.0f);
		vdh23.position = vec3((_v2.position.x + _v3.position.x) / 2.0f, (_v2.position.y + _v3.position.y) / 2.0f, (_v2.position.z + _v3.position.z) / 2.0f);
		vdh31.position = vec3((_v3.position.x + _v1.position.x) / 2.0f, (_v3.position.y + _v1.position.y) / 2.0f, (_v3.position.z + _v1.position.z) / 2.0f);

		VertexData ct1;
		ct1.position = vec3((vdh12.position.x + vdh23.position.x + vdh31.position.x) / 2.0f, (vdh12.position.y + vdh23.position.y + vdh31.position.y) / 2.0f, (vdh12.position.z + vdh23.position.z + vdh31.position.z) / 2.0f);
		ct1.position = ct1.position + height * normal;

		vec3 face1Normal = normalize(cross(vdh12.position - ct1.position, vdh23.position - ct1.position));
		vec3 face2Normal = normalize(cross(vdh12.position - ct1.position, vdh31.position - ct1.position));
		vec3 face3Normal = normalize(cross(vdh23.position - ct1.position, vdh31.position - ct1.position));

		VertexData v1, v2, v3, v4, v5, v6, v7, v8, v9;
		
		v1.position = vdh12.position;
		v2.position = vdh23.position;
		v3.position = ct1.position;
		
		v4.position = vdh12.position;
		v5.position = vdh31.position;
		v6.position = ct1.position;
		
		v7.position = vdh23.position;
		v8.position = vdh31.position;
		v9.position = ct1.position;

		v1.normal = v2.normal = v3.normal = face1Normal;
		v4.normal = v5.normal = v6.normal = face2Normal;
		v7.normal = v8.normal = v9.normal = face3Normal;

		result.push_back(v1);
		result.push_back(v2);
		result.push_back(v3);
		
		result.push_back(v4);
		result.push_back(v5);
		result.push_back(v6);
		
		result.push_back(v7);
		result.push_back(v8);
		result.push_back(v9);
		
		return result;
	}

	void Redraw() {
		tStart = gtstart;
		tEnd = gtend;
		if (hvkLevel > 0) {
			vtxData.clear();
			Build();
		}
	}
};

class Tractricoid : public ParamSurface {
	float tStart, tEnd;
	const float height = 3.0f;
public:
	Tractricoid() { create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		Clifford U(u * height, 1);
		Clifford V(v * 2.0f * M_PI, 0);
		Clifford X = Cos(V) / Cosh(U);
		Clifford Y = Sin(V) / Cosh(U);
		Clifford Z = U - Tanh(U);
		vd.position = vec3(X.f, Y.f, Z.f);

		vec3 drdU = vec3(X.d, Y.d, Z.d);
		U.d = 0; V.d = 1;
		X = Cos(V) / Cosh(U);
		Y = Sin(V) / Cosh(U);
		Z = U - Tanh(U);
		vec3 drdV = vec3(X.d, Y.d, Z.d);

		vd.normal = cross(drdU, drdV);

		vd.texcoord = vec2(u, v);
		return vd;
	}

	void Redraw() {
		vtxData.clear();
		create();
	}

};

//---------------------------
struct Object {
	//---------------------------
	Shader *   shader;
	Material * material;
	Texture *  texture;
	Geometry * geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}
	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) {}

	virtual void Move(){}
};

//Kvaternió forrás: Szirmay-Kalos László, Antal György, Csonka Ferenc: Háromdimenziós grafika, animáció és játékfejlesztés
class Quaternion {             
	float s;	
	vec3 d;	
public:
	Quaternion() : d(0, 0, 1) { s = 0; }
	Quaternion(float s0, vec3 d0) : d(d0) { s = s0; }

	Quaternion operator*(float f) {		
		return Quaternion(s * f, d * f);
	}

	float operator*(Quaternion& q) {		
		return (s * s + dot(d, d));
	}

	void Normalize() {						
		float length = (*this) * (*this);
		(*this) = (*this) * (1 / sqrt(length));
	}


	float GetRotationAngle() {				
		float cosa2 = s;
		float sina2 = length(d);
		float angRad = atan2(sina2, cosa2) * 2;
		return angRad * 180 / M_PI;
	}

	vec3& GetAxis() { return d; }		
};

Quaternion q = Quaternion();

class Virus : public Object {
public:
	Virus(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) : Object(_shader, _material, _texture, _geometry) {
		
	}

	void Animate(float tstart, float tend) {
		if (!isVirusAlive) return;

		rotationAngle = 0.9f * tend;
		float t = tend;
		q = Quaternion(cosf(t/2 * 2 * M_PI), vec3(sinf(t/ 2 * 2 * M_PI), sinf(t/ 3 * 2 * M_PI), sinf(t/ 5 * 2 * M_PI)));
		q.Normalize();
		vec3 axis = q.GetAxis();
		translation = vec3(axis.x, axis.y, axis.z);
	}
};

class AntiVirus : public Object {
public:
	AntiVirus(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) : Object(_shader, _material, _texture, _geometry) {

	}

	void Animate(float tstart, float tend) {
		rotationAngle = 0.8f * tend;
	}

	void Move(vec3 direction) {
		vec3 currentPos = translation;
		vec3 newPos = currentPos + 0.1f * direction;
		translation = newPos;
	}
};

//Forras: https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
mat4 inverse(const mat4& m)
{
	mat4 inverse;

	float A2323 = m[2][2] * m[3][3] - m[2][3] * m[3][2];
	float A1323 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
	float A1223 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
	float A0323 = m[2][0] * m[3][3] - m[2][3] * m[3][0];
	float A0223 = m[2][0] * m[3][2] - m[2][2] * m[3][0];
	float A0123 = m[2][0] * m[3][1] - m[2][1] * m[3][0];
	float A2313 = m[1][2] * m[3][3] - m[1][3] * m[3][2];
	float A1313 = m[1][1] * m[3][3] - m[1][3] * m[3][1];
	float A1213 = m[1][1] * m[3][2] - m[1][2] * m[3][1];
	float A2312 = m[1][2] * m[2][3] - m[1][3] * m[2][2];
	float A1312 = m[1][1] * m[2][3] - m[1][3] * m[2][1];
	float A1212 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
	float A0313 = m[1][0] * m[3][3] - m[1][3] * m[3][0];
	float A0213 = m[1][0] * m[3][2] - m[1][2] * m[3][0];
	float A0312 = m[1][0] * m[2][3] - m[1][3] * m[2][0];
	float A0212 = m[1][0] * m[2][2] - m[1][2] * m[2][0];
	float A0113 = m[1][0] * m[3][1] - m[1][1] * m[3][0];
	float A0112 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

	float det = m[0][0] * (m[1][1] * A2323 - m[1][2] * A1323 + m[1][3] * A1223)
		- m[0][1] * (m[1][0] * A2323 - m[1][2] * A0323 + m[1][3] * A0223)
		+ m[0][2] * (m[1][0] * A1323 - m[1][1] * A0323 + m[1][3] * A0123)
		- m[0][3] * (m[1][0] * A1223 - m[1][1] * A0223 + m[1][2] * A0123);

	if (det == 0) {
		return inverse;
	}

	det = 1 / det;

	inverse[0][0] = det * (m[1][1] * A2323 - m[1][2] * A1323 + m[1][3] * A1223);
	inverse[0][1] = det * -(m[0][1] * A2323 - m[0][2] * A1323 + m[0][3] * A1223);
	inverse[0][2] = det * (m[0][1] * A2313 - m[0][2] * A1313 + m[0][3] * A1213);
	inverse[0][3] = det * -(m[0][1] * A2312 - m[0][2] * A1312 + m[0][3] * A1212);
	inverse[1][0] = det * -(m[1][0] * A2323 - m[1][2] * A0323 + m[1][3] * A0223);
	inverse[1][1] = det * (m[0][0] * A2323 - m[0][2] * A0323 + m[0][3] * A0223);
	inverse[1][2] = det * -(m[0][0] * A2313 - m[0][2] * A0313 + m[0][3] * A0213);
	inverse[1][3] = det * (m[0][0] * A2312 - m[0][2] * A0312 + m[0][3] * A0212);
	inverse[2][0] = det * (m[1][0] * A1323 - m[1][1] * A0323 + m[1][3] * A0123);
	inverse[2][1] = det * -(m[0][0] * A1323 - m[0][1] * A0323 + m[0][3] * A0123);
	inverse[2][2] = det * (m[0][0] * A1313 - m[0][1] * A0313 + m[0][3] * A0113);
	inverse[2][3] = det * -(m[0][0] * A1312 - m[0][1] * A0312 + m[0][3] * A0112);
	inverse[3][0] = det * -(m[1][0] * A1223 - m[1][1] * A0223 + m[1][2] * A0123);
	inverse[3][1] = det * (m[0][0] * A1223 - m[0][1] * A0223 + m[0][2] * A0123);
	inverse[3][2] = det * -(m[0][0] * A1213 - m[0][1] * A0213 + m[0][2] * A0113);
	inverse[3][3] = det * (m[0][0] * A1212 - m[0][1] * A0212 + m[0][2] * A0112);

	return inverse;
}

class CoronaOnVirus : public Object {
	Object *virus;
	VertexData coordinates;
	mat4 locationOnSurface;
	mat4 inverseLocationOnSurface;
public:
	CoronaOnVirus(Object *_virus, Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) : Object(_shader, _material, _texture, _geometry) {
		virus = _virus;
		coordinates.position = vec3(0, 0, 0);
	}

	void Build(VertexData _coords) {
		coordinates = _coords;
		
		vec3 pivot = coordinates.position - 0.2f * coordinates.normal;
		locationOnSurface = mat4(vec4(coordinates.rU.x, coordinates.rU.y, coordinates.rU.z, 0),
			vec4(coordinates.rV.x, coordinates.rV.y, coordinates.rV.z, 0),
			vec4(coordinates.normal.x, coordinates.normal.y, coordinates.normal.z, 0),
			vec4(pivot.x, pivot.y, pivot.z, 1));

		inverseLocationOnSurface = inverse(locationOnSurface);
	}

	void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * locationOnSurface * RotationMatrix(virus->rotationAngle, virus->rotationAxis)*TranslateMatrix(virus->translation);
		Minv = TranslateMatrix(-virus->translation) * RotationMatrix(-virus->rotationAngle, virus->rotationAxis) * inverseLocationOnSurface * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}
};

float rndmove() { return (float)rand() / RAND_MAX; }
enum Direction{FORWARD,BACKWARD,LEFT,RIGHT,UP,DOWN,NONE};
Direction dir = NONE;

class CollisionDetector {
public:
	Object *virus;
	Object *antivirus;
	float virusRadius;
	float antiVirusRadius;

	CollisionDetector(Object *_virus, Object *_antivirus, float _virusRadius, float _antiVirusRadius) {
		virus = _virus;
		antivirus = _antivirus;
		virusRadius = _virusRadius;
		antiVirusRadius = _antiVirusRadius;
	}

	void Detect() {
		vec3 virusCp = virus->translation;
		vec3 antiVirusCp = antivirus->translation;
		float distance = sqrtf(powf(virusCp.x - antiVirusCp.x, 2) + powf(virusCp.y - antiVirusCp.y, 2) + powf(virusCp.z - antiVirusCp.z, 2));

		if (distance < virusRadius + antiVirusRadius) isVirusAlive = false;
	}
};

//---------------------------
class Scene {
	//---------------------------
	std::vector<Object *> objects;
	AntiVirus * tetrahedronObject;
	Virus * sphereObject1;
	Camera camera; // 3D camera
	std::vector<Light> lights;
	CollisionDetector *collisiondetector;
public:
	void Build() {
		// Shaders
		Shader * phongShader = new PhongShader();
		Shader * gouraudShader = new GouraudShader();
		Shader * nprShader = new NPRShader();

		// Materials
		Material * material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		Material * material1 = new Material;
		material1->kd = vec3(0.8f, 0.6f, 0.4f);
		material1->ks = vec3(0.3f, 0.3f, 0.3f);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 30;

		// Textures
		Texture * texture4x8 = new CheckerBoardTexture(4, 8);
		Texture * texture15x20 = new CheckerBoardTexture(45, 50);
		Texture * redTexture = new SimpleRedTexture(4, 8);
		Texture * worldTexture = new worldSphereTexture(4, 8);

		// Geometries
		Geometry * sphere = new Sphere();
		Geometry * tractri = new Tractricoid();
		Geometry * tetrahedron = new Tetrahedron();
		Geometry * world = new Sphere();

		tetrahedronObject = new AntiVirus(phongShader, material0, redTexture, tetrahedron);
		tetrahedronObject->translation = vec3(-7.0f, -2.0f, 0.0f);
		tetrahedronObject->rotationAxis = vec3(0, 1, 1);
		tetrahedronObject->scale = vec3(0.5f, 0.5f, 0.5f);
		objects.push_back(tetrahedronObject);

		Object *sphereObject2 = new Object(phongShader, material0, worldTexture, world);
		sphereObject2->translation = vec3(0.0f, 0.0f, 0.0f);
		sphereObject2->rotationAxis = vec3(1, 0, 0);
		sphereObject2->scale = vec3(20.0f, 20.0f, 20.0f);
		objects.push_back(sphereObject2);

		// Create objects by setting up their vertex data on the GPU
		sphereObject1 = new Virus(phongShader, material0, texture15x20, sphere);
		sphereObject1->translation = vec3(0.0f, 0.0f, 0.0f); 
		sphereObject1->rotationAxis = vec3(1, 1, 0);
		sphereObject1->scale = vec3(1.0f, 1.0f, 1.0f);
		objects.push_back(sphereObject1);

		collisiondetector = new CollisionDetector(sphereObject1, tetrahedronObject, 1.0f, 0.5f);

		std::vector<VertexData> tractricoids = *sphere->getChildrenVtxData();
		for (int i = 0; i < tractricoids.size(); i++) {
			CoronaOnVirus *tractriObject = new CoronaOnVirus(sphereObject1, phongShader, material0, redTexture, tractri);
			tractriObject->scale = vec3(0.1f, 0.1f, 0.1f);
			tractriObject->Build(tractricoids[i]);
			objects.push_back(tractriObject);
		}

		// Camera
		camera.wEye = vec3(0, 0, 8);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		// Lights
		lights.resize(3);
		lights[0].wLightPos = vec4(10, 10, 8, 1);
		lights[0].La = vec3(0.1f, 0.1f, 1);
		lights[0].Le = vec3(1, 0, 0);

		lights[1].wLightPos = vec4(5, 10, 20, 1);
		lights[1].La = vec3(0.2f, 0.2f, 0.2f);
		lights[1].Le = vec3(0, 1, 0);

		lights[2].wLightPos = vec4(-10, 10, 10, 1);
		lights[2].La = vec3(0.1f, 0.1f, 0.1f);
		lights[2].Le = vec3(0, 0, 1);
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object * obj : objects) {
			obj->geometry->Redraw();
			obj->Draw(state);
		}
	}

	void Animate(float tstart, float tend) {
		camera.Animate(tend);
		for (unsigned int i = 0; i < lights.size(); i++) { lights[i].Animate(tend); }
		for (Object * obj : objects) obj->Animate(tstart, tend);
		collisiondetector->Detect();
	}

	void MoveAntiVirus() {
		vec3 direction = vec3(0, 0, 0);
		float rand = rndmove();
		if (rand <= 1.0f / 6.0f || dir == RIGHT) {
			direction = vec3(1, 0, 0);
		}
		else if ((rand > 1.0f / 6.0f && rand <= 2.0f / 6.0f) || dir == LEFT) {
			direction = vec3(-1, 0, 0);
		}
		else if ((rand > 2.0f / 6.0f && rand <= 3.0f / 6.0f) || dir == UP) {
			direction = vec3(0, 1, 0);
		}
		else if ((rand > 3.0f / 6.0f && rand <= 4.0f / 6.0f) || dir == DOWN) {
			direction = vec3(0, -1, 0);
		}
		else if ((rand > 4.0f / 6.0f && rand <= 5.0f / 6.0f) || dir == FORWARD) {
			direction = vec3(0, 0, 1);
		}
		else if ((rand > 5.0f / 6.0f && rand <= 6.0f / 6.0f) || dir == BACKWARD) {
			direction = vec3(0, 0, -1);
		}
		tetrahedronObject->Move(direction);
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	float rand = rndmove();
	float limit = 0.8f;
	if (key == 'x') {
		if (rand <= limit) dir = RIGHT;
		else dir = NONE;
	}
	if (key == 'X') {
		if (rand <= limit) dir = LEFT;
		else dir = NONE;
	}
	if (key == 'y') {
		if (rand <= limit) dir = UP;
		else dir = NONE;
	}
	if (key == 'Y') {
		if (rand <= limit) dir = DOWN;
		else dir = NONE;
	}
	if (key == 'z') {
		if (rand <= limit) dir = FORWARD;
		else dir = NONE;
	}
	if (key == 'Z') {
		if (rand <= limit) dir = BACKWARD;
		else dir = NONE;
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { 
	dir = NONE;
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { 
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

long oldTime = 0;
long timeSum = 0;

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long newTime = glutGet(GLUT_ELAPSED_TIME);
	timeSum += newTime - oldTime;
	if (timeSum >= 100) {
		scene.MoveAntiVirus();
		timeSum = 0;
	}
	oldTime = newTime;

	static float tend = 0;
	const float dt = 0.1f; // dt is ”infinitesimal”
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		gtstart = t;
		gtend = t + Dt;
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}
