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

const float epsilon = 0.0001f;

enum MaterialType { ROUGH, REFLECTIVE };
struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;
	Material(MaterialType t) { type = t;  }
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material * material;
	bool isOnHouse = false;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material * material;
public:
	bool isHouse = false;
	virtual Hit intersect(const Ray& ray) = 0;
};

std::vector<Intersectable *> objects;

struct Disk : public Intersectable {
	vec3 normal;
	vec3 center;
	float radius;

	Disk(const vec3& _normal, Material* mat, const vec3& _center, float _radius) {
		normal = normalize(_normal);
		material = mat;
		center = _center;
		radius = _radius;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		double NdotV = dot(normal, ray.dir);
		if (fabs(NdotV) < epsilon) return hit;
		double t = dot(normal, center - ray.start) / NdotV;
		if (t < epsilon) return hit;
		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		if (sqrtf(powf(hit.position.x - center.x, 2) + powf(hit.position.y - center.y, 2)) > radius) {
			hit.t = -1;
			return hit;
		}
		hit.normal = normal;
		if (dot(hit.normal, ray.dir) > 0) hit.normal = hit.normal * (-1);
		hit.material = material;
		return hit;
	}
};
struct Cylinder : public Intersectable {
	vec3 center;
	float radius;
	bool capped;
	vec2 boundaries;
	vec2 ab;
	float a_d, b_d;

	Cylinder(const vec3& _center, const vec2& _ab, float _radius, Material* _material, bool _capped, vec2 _boundaries) {
		center = _center;
		radius = _radius;
		material = _material;
		capped = _capped;
		boundaries = _boundaries;
		ab = _ab;
		a_d = ab.x;
		b_d = ab.y;
		if (capped) {
			objects.push_back(new Disk(vec3(0, 0, 1), material, vec3(center.x, center.y, boundaries.x), a_d));
			objects.push_back(new Disk(vec3(0, 0, -1), material, vec3(center.x, center.y, boundaries.y), a_d));
		}
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;

		float aa = a_d * a_d;
		float bb = b_d * b_d;

		float a = ray.dir.x * ray.dir.x * bb + ray.dir.y * ray.dir.y * aa;
		float b = 2.0f * (dist.x * ray.dir.x * bb + dist.y * ray.dir.y * aa);
		float c = dist.x * dist.x * bb + dist.y * dist.y * aa - aa * bb;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;

		if (capped) {
			vec3 normal(0, 0, 1);
			vec3 normal2 = vec3(0, 0, -1);

			vec3 point1(center.x, center.y, boundaries.x);
			vec3 point2 = vec3(center.x, center.y, boundaries.y);

			if (dot(normal, (point1 - hit.position)) < 0 || dot(normal, (point2 - hit.position)) > 0) {
				hit.t = -1;
				return hit;
			}
		}

		hit.normal.x = (dist.x + ray.dir.x * hit.t) * 2 * (1 / aa);
		hit.normal.y = (dist.y + ray.dir.y * hit.t) * 2 * (1 / bb);
		hit.normal.z = 0;
		hit.normal = normalize(hit.normal);
		hit.material = material;
		return hit;
	}
};

struct Ellipsoid : public Intersectable {
	vec3 center;
	vec3 abc;
	float a_d, b_d, c_d;
	float boundary;

	Ellipsoid(const vec3& _center, const vec3& _abc, Material* _material, float _boundary) {
		center = _center;
		abc = _abc;
		material = _material;
		boundary = _boundary;
		isHouse = true;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;

		float aa = powf(abc.x, 2);
		float bb = powf(abc.y, 2);
		float cc = powf(abc.z, 2);

		float a = ray.dir.x * ray.dir.x * bb * cc + ray.dir.y * ray.dir.y * aa * cc + ray.dir.z * ray.dir.z * aa * bb;
		float b = 2.0f * (dist.x * ray.dir.x * bb * cc + dist.y * ray.dir.y * aa * cc + dist.z * ray.dir.z * aa * bb);
		float c = dist.x * dist.x * bb * cc + dist.y * dist.y * aa * cc + dist.z * dist.z * aa * bb - aa * bb*cc;

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;

		vec3 normal(0, 0, 1);

		vec3 point1(center.x, center.y, boundary);

		if (dot(normal, (point1 - hit.position)) < 0) {
			hit.t = -1;
			return hit;
		}

		hit.normal.x = -(dist.x + ray.dir.x * hit.t) * 2 * (1 / aa);
		hit.normal.y = -(dist.y + ray.dir.y * hit.t) * 2 * (1 / bb);
		hit.normal.z = -(dist.z + ray.dir.z * hit.t) * 2 * (1 / cc);
		hit.normal = normalize(hit.normal);
		hit.material = material;
		hit.isOnHouse = true;
		return hit;
	}
};

struct Hyperboloid : public Intersectable {
	vec3 center;
	vec3 abc;
	float a_d, b_d, c_d;
	vec2 boundaries;
	bool isTube;

	Hyperboloid(const vec3& _center, const vec3& _abc, Material* _material, const vec2& _boundaries, bool _isTube = false) {
		center = _center;
		abc = _abc;
		material = _material;
		boundaries = _boundaries;
		a_d = abc.x;
		b_d = abc.y;
		c_d = abc.z;
		isTube = _isTube;
		isHouse = isTube ? true : false;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;

		float aa = powf(a_d, 2);
		float bb = powf(b_d, 2);
		float cc = powf(c_d, 2);

		float a = ray.dir.x * ray.dir.x * bb * cc + ray.dir.y * ray.dir.y * aa * cc - ray.dir.z * ray.dir.z * aa * bb;
		float b = 2.0f * (dist.x * ray.dir.x * bb * cc + dist.y * ray.dir.y * aa * cc - dist.z * ray.dir.z * aa * bb);
		float c = dist.x * dist.x * bb * cc + dist.y * dist.y * aa * cc - dist.z * dist.z * aa * bb - aa * bb*cc;

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;

		if (t1 > 0 && t2 > 0) {
			vec3 pos1 = ray.start + ray.dir * t2;
			vec3 pos2 = ray.start + ray.dir * t1;

			vec3 normal(0, 0, 1);
			vec3 point1(center.x, center.y, boundaries.x);
			vec3 normal2 = vec3(0, 0, -1);
			vec3 point2 = vec3(center.x, center.y, boundaries.y);

			if ((dot(normal, (point1 - pos1)) < 0 && dot(normal, (point1 - pos2)) < 0) || (dot(normal2, (point2 - pos1)) < 0 && dot(normal2, (point2 - pos2)) < 0)) { //kisebb nagyobb  || dot(normal, (point2 - hit.position)) > 0
				hit.t = -1;
				return hit;
			}
			else if ((dot(normal, (point1 - pos1)) < 0 && dot(normal, (point1 - pos2)) >= 0) || (dot(normal2, (point2 - pos1)) < 0 && dot(normal2, (point2 - pos2)) >= 0)) {
				hit.t = t1;
			}
			else {
				hit.t = t2;
			}
		}

		hit.position = ray.start + ray.dir * hit.t;
		if (isTube) {
			hit.normal.x = (-1) * (dist.x + ray.dir.x * hit.t) * 2 * (1 / aa);
			hit.normal.y = (-1) * (dist.y + ray.dir.y * hit.t) * 2 * (1 / bb);
			hit.normal.z = (dist.z + ray.dir.z * hit.t) * 2 * (1 / cc);
			hit.normal = normalize(hit.normal);
		}
		else {
			hit.normal.x = (dist.x + ray.dir.x * hit.t) * 2 * (1 / aa);
			hit.normal.y = (dist.y + ray.dir.y * hit.t) * 2 * (1 / bb);
			hit.normal.z = -(dist.z + ray.dir.z * hit.t) * 2 * (1 / cc);
			hit.normal = normalize(hit.normal);
		}
		hit.material = material;
		return hit;
	}
};

struct Paraboloid : public Intersectable {
	vec3 center;
	vec2 ab;
	float a_d, b_d;
	bool capped;
	float boundary;

	Paraboloid(const vec3& _center, const vec2& _ab, Material* _material, bool _capped, float _boundary) {
		center = _center;
		ab = _ab;
		material = _material;
		a_d = ab.x;
		b_d = ab.y;
		capped = _capped;
		boundary = _boundary;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;

		float aa = powf(a_d, 2);
		float bb = powf(b_d, 2);

		float a = ray.dir.x * ray.dir.x * bb + ray.dir.y * ray.dir.y * aa;
		float b = 2.0f * dist.x * ray.dir.x * bb + 2.0f * dist.y * ray.dir.y * aa + ray.dir.z * aa * bb;
		float c = dist.x * dist.x * bb + dist.y * dist.y * aa + dist.z * aa * bb;

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;

		if (capped) {
			vec3 normal(0, 0, -1);

			vec3 point1(center.x, center.y, boundary);

			if (dot(normal, (point1 - hit.position)) < 0) {
				hit.t = -1;
				return hit;
			}
		}

		hit.normal.x = (dist.x + ray.dir.x * hit.t) * 2 * (1 / aa);
		hit.normal.y = (dist.y + ray.dir.y * hit.t) * 2 * (1 / bb);
		hit.normal.z = (dist.z + ray.dir.z * hit.t) * -2;
		hit.normal = normalize(hit.normal);
		hit.material = material;
		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

class Scene {
	std::vector<Light *> lights;
	std::vector<vec3> topPoints;
	Camera camera;
	vec3 La;
	vec3 sky;
	vec3 sun;
	vec3 sunDir;
	vec3 Le;
	float radius = 0.4f;
	vec3 top;

public:
	void build() {
		vec3 eye = vec3(0, 1.8, 0), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0);
		float fov = 100 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.1f, 0.1f, 0.1f);
		sky = vec3(0.97f, 0.97f, 0.99f);
		sun = vec3(0.2f, 0.2f, 0.1f);
		sunDir = vec3(-0.5f, -0.5f, -0.5f);
		Le = vec3(0.07, 0.07, 0.07);
		top = vec3(0.0f, 0.0f, 0.98f);

		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		vec3 kd2(0.8f, 0.5f, 0.1f), ks2(2, 2, 2);
		vec3 n(0.17f, 0.35f, 1.5f), kappa(3.1f, 2.7f, 1.9f);
		vec3 n2(0.14f, 0.16f, 0.13f), kappa2(4.1f, 2.3f, 3.1f);
		Material * material1 = new RoughMaterial(kd, ks, 800);
		Material * material2 = new ReflectiveMaterial(n, kappa);
		Material * material3 = new RoughMaterial(vec3(0.1f, 0.5f, 0.2f), vec3(3, 3, 3), 15);
		Material * material4 = new ReflectiveMaterial(n2, kappa2);
		Material * material5 = new RoughMaterial(kd2, ks2, 15);

		objects.push_back(new Ellipsoid(vec3(0.0f, 0.0f, 0.0f), vec3(2.0f, 2.0f, 1.0f), material1, 0.98));
		objects.push_back(new Paraboloid(vec3(-0.8f, 0.0f, 0.0f), vec2(0.7f, 0.7f), material2, true, -1.0f));
		objects.push_back(new Hyperboloid(vec3(0.75f, 0.6f, 0.0f), vec3(0.1f, 0.1f, 0.3f), material3, vec2(0.3f, -1.5f), false)); 
		objects.push_back(new Cylinder(vec3(0.4f, -0.6f, 0.0f), vec2(0.35f, 0.35f), 1.0f, material5, true, vec2(-0.3f, -1.2f)));
		objects.push_back(new Hyperboloid(vec3(0.0f, 0.0f, 0.0f), vec3(0.09f, 0.09f, 0.2f), material4, vec2(2.0f, 0.98f), true));
		topPoints = createHole(0.4f, 0.98f);
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {
		for (Intersectable * object : objects) if (object->intersect(ray).t > epsilon && object->intersect(ray).position.z < top.z) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		vec3 outRadiance(0, 0, 0);

		if (depth > 5) return La;

		if (hit.t < 0) return sky + sun * powf(dot(normalize(ray.dir), normalize(sunDir)), 10);

		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La;
			vec3 normal = hit.normal;
			vec3 outDir;
			float A = powf(radius, 2) * M_PI;
			float n = float(topPoints.size());
			for (int i = 0; i < n; i++) {
				vec3 hitPoint = hit.position + hit.normal * epsilon;
				outDir = normalize(topPoints[i] - hitPoint);
				vec3 shadowDir = normalize(topPoints[i] - hitPoint);
				Ray shadowRay(hitPoint, shadowDir);
				float cosTheta = dot(hit.normal, outDir);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
					float cosT = dot(normalize(outDir), vec3(0.0f, 0.0f, -1.0f));
					float rN = powf(hitPoint.x - topPoints[i].x, 2) + powf(hitPoint.y - topPoints[i].y, 2) + powf(hitPoint.z - topPoints[i].z, 2);
					float deltaW = (A / n) * (cosT / rN);
					outRadiance = outRadiance + Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + normalize(outDir));
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
					outRadiance = outRadiance + trace(Ray(hitPoint, outDir), depth + 1) * deltaW;
				}
			}
		}
		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}
		return outRadiance;
	}

	std::vector<vec3> createHole(float radius, float height) {
		float a = radius;
		std::vector<vec3> result;

		for (float i = -a; i <= a; i += 0.1f) {
			for (float j = -a; j <= a; j += 0.1f) {
				if (sqrtf(powf(i, 2) + powf(j, 2)) <= radius) {
					result.push_back(vec3(i, j, height));
				}
			}
		}
		return result;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
}