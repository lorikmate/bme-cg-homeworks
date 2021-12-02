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

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU
unsigned int vaoPoints;

//Forras: https://noobtuts.com/cpp/compare-float-values
bool cmpf(float A, float B, const float epsilon = 1e-8f)
{
	return (fabs(A - B) < epsilon);
}

class Circle {
private:
	float radius;
	float points[100];
	int segments;
	float cx = 0;
	float cy = 0;
public:

	Circle() {}

	Circle(float r, float x, float y) {
		radius = r;
		cx = x;
		cy = y;
		segments = 100;

		for (int i = 0; i < segments; i++) {
			points[i] = 0;
		}
	}

	void createCircle() {
		for (int i = 0; i < segments; i += 2)
		{
			float phi = 2.0f * M_PI * float(i) / float(segments);

			float x = cosf(phi);
			float y = sinf(phi);

			points[i] = x + cx;
			points[i + 1] = y + cy;
		}
	}

	bool isPointOnCircle(float px, float py) {
		return (sqrtf(powf(px, 2) + powf(py, 2)) <= radius);
	}

	float *getPoints() {
		return points;
	}

	float getRadius() {
		return radius;
	}

	int getSegments() {
		return segments;
	}
};

class Triangle {
private:
	std::vector<vec4> vertexes;
	std::vector<vec4> generatedVertices;
	std::vector<vec4> tesselation;
	std::vector<vec4> outLine;
	vec3 c;
	vec3 c2;
	vec3 c3;
	float r, r2, r3;

public:
	Triangle(){}

	Triangle(vec2 pt1, vec2 pt2, vec2 pt3){
		vertexes.push_back(vec4(pt1.x, pt1.y, 0, 1));
		vertexes.push_back(vec4(pt2.x, pt2.y, 0, 1));
		vertexes.push_back(vec4(pt3.x, pt3.y, 0, 1));
	}

	void addPoint(vec4 point) {
		vertexes.push_back(point);
		if (vertexes.size() == 3) {
			vec4 p1 = vertexes[0];
			vec4 p2 = vertexes[1];
			vec4 p3 = vertexes[2];

			c = calculateCenter(p1, p2);
			r = calculateRadius(c, p1);

			c2 = calculateCenter(p2, p3);
			r2 = calculateRadius(c2, p2);

			c3 = calculateCenter(p3, p1);
			r3 = calculateRadius(c3, p1);

			generatedVertices = generateArc(p1, p2, c, r);

			std::vector<vec4> generatedVertices2 = generateArc(p2, p3, c2, r2);
			std::vector<vec4> generatedVertices3 = generateArc(p3, p1, c3, r3);

			float B = calculateLength(generatedVertices);
			float A = calculateLength(generatedVertices2);
			float C = calculateLength(generatedVertices3);

			generatedVertices.insert(generatedVertices.end(), generatedVertices2.begin(), generatedVertices2.end());
			generatedVertices.insert(generatedVertices.end(), generatedVertices3.begin(), generatedVertices3.end());

			float Alpha = calculateAngle(c2, c3, p3) * 180 / M_PI;
			float Beta = calculateAngle(c, c2, p2) * 180 / M_PI;
			float Gamma = calculateAngle(c, c3, p1) * 180 / M_PI;
			float Sum = Alpha + Beta + Gamma;

			printf("\nAlpha: %.6f, Beta: %.6f, Gamma: %.6f, Angle sum: %.6f", Alpha, Beta, Gamma, Sum);
			printf("\na: %.6f, b: %.6f, c: %.6f", A, B, C);

			for (int i = 0; i < generatedVertices.size(); i++)
				outLine.push_back(vec4(generatedVertices[i].x, generatedVertices[i].y, 0, 1));

			tesselation = triangulate(generatedVertices);
		}
	}

	vec3 calculateCenter(vec4 p1, vec4 p2) {
		float pfx = (p1.x + p2.x) / 2;
		float pfy = (p1.y + p2.y) / 2;
		float pmx = p1.x - p2.x;
		float pmy = p1.y - p2.y;

		float cX = (-1)*(p1.y * pmx - pmy * p1.x + powf(p1.x, 2) * p1.y * pmx - pmy * powf(p1.x, 3) + powf(p1.y, 3) * pmx - pmy * p1.x * powf(p1.y, 2) + 2 * pmy * p1.x * p1.y * pfy + 2 * p1.x * p1.y * pfx * pmx - p1.y * pmx - powf(p1.x, 2) * p1.y * pmx - p1.y * powf(p1.y, 2) * pmx) / (2 * p1.x*((-1)*p1.y * pmx + pmy * p1.x));
		float cY = (2 * p1.x*pfy*pmy + 2 * p1.x*pfx*pmx - pmx - powf(p1.x, 2) * pmx - pmx * powf(p1.y, 2)) / (2 * ((-1) * pmx * p1.y + p1.x * pmy));

		return vec3(cX, cY, 0);
	}

	float calculateRadius(vec3 center, vec4 p1) {
		return length(vec3(p1.x, p1.y, 0) - center);
	}

	//Forras: http://www.cplusplus.com/reference/algorithm/reverse/
	template<class Iter>
	void reverse(Iter first, Iter last) {
		while ((first != last) && (first != --last)) {
			std::iter_swap(first, last);
			++first;
		}
	}

	std::vector<vec4> generateArc(vec4 p1, vec4 p2, vec3 c, float r) {
		std::vector<vec4> arc;

		float theta = atan2(p1.y - c.y, p1.x - c.x);
		float phi = atan2(p2.y - c.y, p2.x - c.x);

		float d_theta = (theta <= 0) ? (2 * M_PI + theta) : theta;
		float d_phi = (phi <= 0) ? (2 * M_PI + phi) : phi;

		float step = M_PI/512;

		if (d_theta < d_phi && d_phi - d_theta < M_PI) {
			for (float a = d_theta; a <= d_phi; a += step)
			{
				float x = cosf(a) * r;
				float y = sinf(a) * r;
				arc.push_back(vec4(x + c.x, y + c.y, 0, 1));
			}
		}
		else if (d_phi < d_theta && d_theta - d_phi < M_PI) {
			for (float a = d_phi; a <= d_theta; a += step)
			{
				float x = cosf(a) * r;
				float y = sinf(a) * r;
				arc.push_back(vec4(x + c.x, y + c.y, 0, 1));
			}
			reverse(arc.begin(), arc.end());
		}
		else if (d_theta < d_phi && d_phi - d_theta > M_PI) {
			for (float a = phi; a <= theta; a += step)
			{
				float x = cosf(a) * r;
				float y = sinf(a) * r;
				arc.push_back(vec4(x + c.x, y + c.y, 0, 1));
			}
			reverse(arc.begin(), arc.end());
		}
		else if (d_phi < d_theta && d_theta - d_phi > M_PI) {
			for (float a = theta; a <= phi; a += step)
			{
				float x = cosf(a) * r;
				float y = sinf(a) * r;
				arc.push_back(vec4(x + c.x, y + c.y, 0, 1));
			}
		}
		return arc;
	}

	float calculateLength(std::vector<vec4> arc) {
		float totalLength = 0;
		for (int i = 0; i < arc.size()-1; i++) {
			float x = arc[i].x;
			float y = arc[i].y;
			float dx = arc[i + 1].x - x;
			float dy = arc[i + 1].y - y;
			
			float ds = sqrtf(dx * dx + dy * dy) / (1 - x * x - y * y);

			totalLength += ds;
		}
		return totalLength;
	}

	float calculateAngle(vec3 c1, vec3 c2, vec4 p) {
		vec3 dirVec1 = vec3(c1.x - p.x, c1.y - p.y, 0);
		vec3 dirVec2 = vec3(c2.x - p.x, c2.y - p.y, 0);
		float angle = acosf(dot(dirVec1,dirVec2) / (length(dirVec1) * length(dirVec2)));

		if (dirVec1.x < 0 && dirVec2.x >= 0 || dirVec1.y < 0 && dirVec2.y >= 0 || dirVec1.x >= 0 && dirVec2.x < 0 || dirVec1.y >= 0 && dirVec2.y < 0)
			angle = M_PI - angle;

		return angle;
	}

	bool lineIntersect(vec4 p1, vec4 p2, vec4 p3, vec4 p4) {
		if ((cmpf(p1.x, p3.x) && cmpf(p1.y, p3.y)) || (cmpf(p1.x, p4.x) && cmpf(p1.y, p4.y)) || (cmpf(p3.x, p2.x) && cmpf(p3.y, p2.y)) || (cmpf(p4.x, p2.x) && cmpf(p4.y, p2.y))) return false;

		vec4 directionVec = vec4(p1.x - p2.x, p1.y - p2.y, 0, 1);
		vec4 normalVec = vec4((-1)*directionVec.y, directionVec.x, 0, 1);
		float result1 = dot(normalVec, (p3 - p1)) * dot(normalVec, (p4 - p1));

		vec4 dirVec = vec4(p3.x - p4.x, p3.y - p4.y);
		vec4 norVec = vec4((-1) * dirVec.y, dirVec.x);
		float result2 = dot(norVec, (p1 - p3)) * dot(norVec, (p2 - p3));

		if (result1 < 0 && result2 < 0) return true;
		else return false;
	}

	bool isEar(std::vector<vec4> polygon, vec4 previous, vec4 current, vec4 next) {
		int intersectCount = 0;
		vec4 midpoint = vec4((previous.x + next.x) / 2, (previous.y + next.y) / 2, 0, 1);
		vec4 rightPoint = vec4(midpoint.x + 20, midpoint.y, 0, 1);

		if (lineIntersect(previous, next, previous, current) || lineIntersect(previous, next, current, next)) {
			return false;
		}

		for (int i = 0; i < polygon.size() - 1; i++) {
			if (lineIntersect(previous, next, polygon[i], polygon[i + 1])) {
				return false;
			}
		}

		if (lineIntersect(previous, next, polygon[0], polygon[polygon.size() - 1])) {
			return false;
		}

		for (int i = 0; i < polygon.size() - 1; i++) {
			if (lineIntersect(midpoint, rightPoint, polygon[i], polygon[i + 1])) {
				intersectCount += 1;
			}
		}

		if (lineIntersect(midpoint, rightPoint, polygon[0], polygon[polygon.size() - 1])) {
			intersectCount += 1;
		}

		if (intersectCount % 2 == 0) {
			return false;
		}

		return true;
	}

	std::vector<vec4> triangulate(std::vector<vec4>& arcArray){
		std::vector<vec4> tessellation;
		while (arcArray.size() > 3) {
			bool flag = false;
			for (int i = 0; i < arcArray.size(); i++) {
				int previousIndex = i - 1 < 0 ? ((i-1) + arcArray.size()) % arcArray.size() : i - 1;
				int nextIndex = i + 1 > 0 ? (i+1) % arcArray.size() : i + 1;
				if (isEar(arcArray, arcArray[previousIndex], arcArray[i], arcArray[nextIndex])) {
					tessellation.push_back(arcArray[previousIndex]);
					tessellation.push_back(arcArray[i]);
					tessellation.push_back(arcArray[nextIndex]);
					arcArray.erase(arcArray.begin() + i);
					flag = true;
					break;
				}
			}
			if (flag == false) {
				break;
			}
		}
		if (arcArray.size() <= 3) {
			for (int i = 0; i < arcArray.size(); i++) {
				tessellation.push_back(arcArray[i]);
			}
			arcArray.clear();
			return tessellation;
		}
		return tessellation;
	}

	std::vector<vec4> getTesselation() {
		return tesselation;
	}

	std::vector<vec4> getVertexes() {
		return vertexes;
	}

	std::vector<vec4> getOutLine() {
		return outLine;
	}
};

Circle *baseCircle;
Triangle *siriusTriangle;

class SiriusGeometry {
private:
	std::vector<vec4> pts;
	unsigned int vbo;		// vertex buffer object
	unsigned int vboPoints;		// vertex buffer object
public:

	SiriusGeometry() {
		//Circle
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed

		//Points
		glGenVertexArrays(1, &vaoPoints);	// get 1 vao id
		glBindVertexArray(vaoPoints);		// make it active
		glGenBuffers(1, &vboPoints);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vboPoints);
		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			4 * sizeof(float), NULL); 		     // stride, offset: tightly packed

		//Create base circle
		baseCircle = new Circle(1.0f, 0.0f, 0.0f);
		baseCircle->createCircle();

		//Create triangle
		siriusTriangle = new Triangle();
	}

	void addPoint(float x, float y) {
		if (baseCircle->isPointOnCircle(x, y)) {
			if (pts.size() < 3) {
				pts.push_back(vec4(x, y, 0, 1));
				siriusTriangle->addPoint(vec4(x, y, 0, 1));
			}
		}
	}

	void draw() {
		//Circle
		glBindVertexArray(vao); 
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, baseCircle->getSegments() * sizeof(float), &baseCircle->getPoints()[0], GL_STATIC_DRAW);
		glDrawArrays(GL_TRIANGLE_FAN, 0, baseCircle->getSegments());

		//Points
		glBindVertexArray(vaoPoints);
		glBindBuffer(GL_ARRAY_BUFFER, vboPoints);
		glBufferData(GL_ARRAY_BUFFER, pts.size() * 4 * sizeof(float), &pts[0], GL_DYNAMIC_DRAW);
		gpuProgram.setUniform(vec3(1, 0, 0), "color");
		glPointSize(6.0f);
		glDrawArrays(GL_POINTS, 0, pts.size());

		//Triangle outline
		if (siriusTriangle->getVertexes().size() == 3) {
			glBufferData(GL_ARRAY_BUFFER, siriusTriangle->getOutLine().size() * 4 * sizeof(float), &siriusTriangle->getOutLine()[0], GL_DYNAMIC_DRAW);
			gpuProgram.setUniform(vec3(0.23f, 0.15f, 0.29f), "color");
			glLineWidth(5.0f);
			glDrawArrays(GL_LINE_LOOP, 0, siriusTriangle->getOutLine().size());
		}

		//Triangle infill
		glBufferData(GL_ARRAY_BUFFER, siriusTriangle->getTesselation().size() * 4 * sizeof(float), &siriusTriangle->getTesselation()[0], GL_DYNAMIC_DRAW);
		gpuProgram.setUniform(vec3(0.99f, 0.76f, 0.71f), "color");
		glDrawArrays(GL_TRIANGLES, 0, siriusTriangle->getTesselation().size());
	}
};

SiriusGeometry *geometry;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	
	geometry = new SiriusGeometry();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
	
	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.5f, 0.5f, 0.5f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	geometry->draw();
	
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		geometry->addPoint(cX, cY);
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
