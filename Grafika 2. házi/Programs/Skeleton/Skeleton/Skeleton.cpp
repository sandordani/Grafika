//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2018. osztol.
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
// Nev    : Sandor Daniel
// Neptun : F193CB
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

const char *vertexSource = R"(
	#version 330
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;

	layout(location = 0) in vec2 cCamWindowVertex;
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";

const char *fragmentSource = R"(
	#version 330
    precision highp float;

	struct Material {
		vec3 ka, kd, ks;
		float  shininess;
		vec3 F0;
		int rough, reflective;
	};

	struct Light {
		vec3 direction;
		vec3 Le, La;
	};

	struct Ellipsoid {
		vec3 center, scale;
		float radius;
	};

	struct Rectangle {
		vec3 a, b, c, d;
	};

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;
	};

	struct Ray {
		vec3 start, dir;
	};

	
	const int maxRectangles = 100;
 
	uniform int gold;
	uniform vec3 wEye; 
	uniform Light light;     
	uniform Material materials[5];
	uniform int nEllipsoids;
	uniform int nRectangles;
	uniform Ellipsoid ellipsoids[3];
	uniform Rectangle rectangles[maxRectangles];

	in  vec3 p;
	out vec4 fragmentColor;


	Hit intersect(const Ellipsoid object, const Ray ray) {
		Hit hit;
		hit.t = -1;

		vec3 scaledStart = ray.start * object.scale;
		vec3 scaledCenter = object.center * object.scale;
		vec3 scaledDir = ray.dir * object.scale;

		vec3 dist = scaledStart - scaledCenter;
		float a = dot(scaledDir, scaledDir);
		float b = dot(dist, scaledDir) * 2.0;
		float c = dot(dist, dist) - object.radius * object.radius;
		float discr = b * b - 4.0 * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0 / a;
		float t2 = (-b - sqrt_discr) / 2.0 / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;

		hit.normal = normalize((hit.position - object.center) * object.scale * 2);
		return hit;
	}

	Hit intersect(const Rectangle object, const Ray ray) {
		Hit hit;
		hit.t = -1;
		
		vec3 n = cross( (object.b-object.a), (object.d-object.a) );
		float t = dot( (object.a-ray.start), n) / dot( ray.dir, n);

		if(t<0) return hit;
		
		vec3 pos=ray.start + ray.dir * t;
		
		if(dot(cross((object.b-object.a),(pos-object.a)),n)<=0) return hit;
		if(dot(cross((object.c-object.b),(pos-object.b)),n)<=0) return hit;	
		if(dot(cross((object.d-object.c),(pos-object.c)),n)<=0) return hit;	
		if(dot(cross((object.a-object.d),(pos-object.d)),n)<=0) return hit;		

		hit.t=t;
		hit.position=pos;
		hit.normal = normalize(n);

		return hit;
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		for (int o = 0; o < nEllipsoids; o++) {
			Hit hit = intersect(ellipsoids[o], ray);
			hit.mat = o;
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		for (int o = 0; o < nRectangles; o++) {
			Hit hit = intersect(rectangles[o], ray);
			if(gold==1)hit.mat = 3;
			else hit.mat = 4;
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {
		for (int o = 0; o < nEllipsoids; o++) if (intersect(ellipsoids[o], ray).t > 0) return true;
		return false;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}

	const float epsilon = 0.0001f;
	const int maxdepth = 10;

	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0, 0, 0);
		for(int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) return weight * light.La;
			if (materials[hit.mat].rough == 1) {
				outRadiance += weight * materials[hit.mat].ka * light.La;
				Ray shadowRay;
				shadowRay.start = hit.position + hit.normal * epsilon;
				shadowRay.dir = light.direction;
				float cosTheta = dot(hit.normal, light.direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance += weight * light.Le * materials[hit.mat].kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light.direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance += weight * light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
				}
			}

			if (materials[hit.mat].reflective == 1) {
				weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			} else return outRadiance;
		}
	}

	void main() {
		Ray ray;
		ray.start = wEye; 
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1); 
	}
)";

class Material {
protected:
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	bool rough, reflective;
public:
	void SetUniform(unsigned int shaderProg, int mat) {
		char buffer[256];
		sprintf(buffer, "materials[%d].ka", mat);
		ka.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].kd", mat);
		kd.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].ks", mat);
		ks.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].shininess", mat);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform material.shininess cannot be set\n");
		sprintf(buffer, "materials[%d].F0", mat);
		F0.SetUniform(shaderProg, buffer);

		sprintf(buffer, "materials[%d].rough", mat);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, rough ? 1 : 0); else printf("uniform material.rough cannot be set\n");
		sprintf(buffer, "materials[%d].reflective", mat);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, reflective ? 1 : 0); else printf("uniform material.reflective cannot be set\n");
	}
};

class RoughMaterial : public Material {
public:
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};

class SmoothMaterial : public Material {
public:
	SmoothMaterial(vec3 _F0) {
		F0 = _F0;
		rough = false;
		reflective = true;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

struct Mirror {
	vec3 a, b, c, d;

	Mirror(const vec3& _a, const vec3& _b, const vec3& _c, const vec3& _d) { a = _a; b = _b; c = _c; d = _d; }
	void SetUniform(unsigned int shaderProg, int o) {
		char buffer[256];
		sprintf(buffer, "rectangles[%d].a", o);
		a.SetUniform(shaderProg, buffer);
		sprintf(buffer, "rectangles[%d].b", o);
		b.SetUniform(shaderProg, buffer);
		sprintf(buffer, "rectangles[%d].c", o);
		c.SetUniform(shaderProg, buffer);
		sprintf(buffer, "rectangles[%d].d", o);
		d.SetUniform(shaderProg, buffer);
	}
};

struct Ellipsoid {
	vec3 center, scale;
	float radius;
	vec2 v = vec2(0.1, 0.1);

	Ellipsoid(const vec3& _center, const vec3& _scale, float _radius) { center = _center; scale = _scale; radius = _radius; }
	void SetUniform(unsigned int shaderProg, int o) {
		char buffer[256];
		sprintf(buffer, "ellipsoids[%d].center", o);
		center.SetUniform(shaderProg, buffer);
		sprintf(buffer, "ellipsoids[%d].scale", o);
		scale.SetUniform(shaderProg, buffer);
		sprintf(buffer, "ellipsoids[%d].radius", o);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, radius);
		else printf("uniform %s cannot be set\n", buffer);
	}

	void Animate(float t, std::vector<Mirror *> mirrors) {
		vec3 oldcenter = center;
		vec2 f = vec2(rnd()-0.5, rnd()-0.5);
		float m = 1;
		vec2 a =f * (1/m);
		v = v + a * t;
		center = vec3(center.x + v.x*t, center.y + v.y*t, center.z);
		for (int i = 0; i < mirrors.size(); i++) {
			vec3 mirrorside = mirrors[i]->c - mirrors[i]->d;
			if (dot(cross(mirrorside, center - mirrors[i]->d), vec3(0, 0, 1)) <= 0) {
				vec2 n = vec2(mirrorside.y, -mirrorside.x);
				float cosa = dot(n, v) / length(n) / length(v);
				float alpha = acosf(cosa);
				v = -v;
				if (dot(mirrorside, v) / length(mirrorside) / length(v) < 0)
					v = vec2(v.x*cosf(2 * alpha) + v.y*sinf(2 * alpha), -v.x*sinf(2 * alpha) + v.y*cosf(2 * alpha));
				else
					v = vec2(v.x*cosf(2 * alpha) - v.y*sinf(2 * alpha), v.x*sinf(2 * alpha) + v.y*cosf(2 * alpha));
				center = vec3(oldcenter.x+v.x*t, oldcenter.y+v.y*t, oldcenter.z);
			}
		}

	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, double _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tan(fov / 2);
		up = normalize(cross(w, right)) * f * tan(fov / 2);
	}

	void SetUniform(unsigned int shaderProg) {
		eye.SetUniform(shaderProg, "wEye");
		lookat.SetUniform(shaderProg, "wLookAt");
		right.SetUniform(shaderProg, "wRight");
		up.SetUniform(shaderProg, "wUp");
	}
};

struct Light {
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
	void SetUniform(unsigned int shaderProg) {
		La.SetUniform(shaderProg, "light.La");
		Le.SetUniform(shaderProg, "light.Le");
		direction.SetUniform(shaderProg, "light.direction");
	}
};



class Scene {
	std::vector<Ellipsoid *> ellipsoids;
	std::vector<Mirror *> rectangles;
	Light* light;
	Camera camera;
	std::vector<Material *> materials;
	int nMirrors = 3;
	float mirrorEnd = -10, eyez;
	bool isGold = true;

public:
	void build() {
		vec3 eye = vec3(0, 0, 2);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		eyez = eye.z;

		light=new Light(vec3(0, 0, 0.1), vec3(1, 1, 1), vec3(0.5, 0.5, 0.5));

		ellipsoids.push_back(new Ellipsoid(vec3(0, 0, mirrorEnd), vec3(0.6, 1, 0.5), 0.1));
		ellipsoids.push_back(new Ellipsoid(vec3(0, 0.3, mirrorEnd), vec3(0.6, 1, 0.5), 0.1));
		ellipsoids.push_back(new Ellipsoid(vec3(0, -0.3, mirrorEnd), vec3(0.6, 1, 0.5), 0.1));

		for (int i = 0; i < nMirrors; i++)rectangles.push_back(new Mirror(vec3(cosf(2 * M_PI*i / nMirrors), sinf(2 * M_PI*i / nMirrors), eyez),
			vec3(cosf(2 * M_PI*(i + 1) / nMirrors), sinf(2 * M_PI*(i + 1) / nMirrors), eyez),
			vec3(cosf(2 * M_PI*(i + 1) / nMirrors), sinf(2 * M_PI*(i + 1) / nMirrors), mirrorEnd),
			vec3(cosf(2 * M_PI*i / nMirrors), sinf(2 * M_PI*i / nMirrors), mirrorEnd)));
		
		vec3 kd(0.5f, 0, 0.5f), ks( 10, 10, 10);
		materials.push_back(new RoughMaterial(kd, ks, 50));
		kd = vec3(0.5f, 0.5f, 0);
		materials.push_back(new RoughMaterial(kd, ks, 50));
		kd = vec3(0, 0.5f, 0.5f);
		materials.push_back(new RoughMaterial(kd, ks, 50));

		vec3 nGold = vec3(0.17f, 0.35f, 1.5f);
		vec3 kGold = vec3(3.1f, 2.7f, 1.9f);
		vec3 nSilver = vec3(0.14f, 0.16f, 0.13f);
		vec3 kSilver = vec3(4.1f, 2.3f, 3.1f);

		vec3 denF0G = (nGold + vec3(1, 1, 1))*(nGold + vec3(1, 1, 1)) + (kGold*kGold);
		vec3 F0G = ((nGold - vec3(1, 1, 1))*(nGold - vec3(1, 1, 1)) + (kGold*kGold)) * vec3(1 / denF0G.x, 1 / denF0G.y, 1 / denF0G.z);
		vec3 denF0S = ((nSilver + vec3(1, 1, 1))*(nSilver + vec3(1, 1, 1)) + (kSilver*kSilver));
		vec3 F0S = ((nSilver - vec3(1, 1, 1))*(nSilver - vec3(1, 1, 1)) + (kSilver*kSilver)) * vec3(1 / denF0S.x, 1 / denF0S.y, 1 / denF0S.z);

		materials.push_back(new SmoothMaterial(F0G));
		materials.push_back(new SmoothMaterial(F0S));

	}

	void SetUniform(unsigned int shaderProg) {
		int location = glGetUniformLocation(shaderProg, "nEllipsoids");
		if (location >= 0) glUniform1i(location, ellipsoids.size()); else printf("uniform nEllipsoids cannot be set\n");
		for (int o = 0; o < ellipsoids.size(); o++) ellipsoids[o]->SetUniform(shaderProg, o);
		location = glGetUniformLocation(shaderProg, "nRectangles");
		if (location >= 0) glUniform1i(location, rectangles.size()); else printf("uniform nRectangles cannot be set\n");
		for (int o = 0; o < rectangles.size(); o++) rectangles[o]->SetUniform(shaderProg, o);
		light->SetUniform(shaderProg);
		camera.SetUniform(shaderProg);
		for (int mat = 0; mat < materials.size(); mat++) materials[mat]->SetUniform(shaderProg, mat);
		location = glGetUniformLocation(shaderProg, "gold");
		if (isGold)
			if (location >= 0) glUniform1i(location, 1); else printf("uniform gold cannot be set\n");
		else
			if (location >= 0) glUniform1i(location, 0); else printf("uniform gold cannot be set\n");
	}
	void Animate(float t) {
		for (int i = 0; i < ellipsoids.size(); i++)
		{
			ellipsoids[i]->Animate(t, rectangles);
		}
	}

	void IncrementMirrors() {
		rectangles.clear();
		nMirrors += 1;
		for (int i = 0; i < nMirrors; i++)rectangles.push_back(new Mirror(vec3(cosf(2 * M_PI*i / nMirrors), sinf(2 * M_PI*i / nMirrors), eyez),
			vec3(cosf(2 * M_PI*(i + 1) / nMirrors), sinf(2 * M_PI*(i + 1) / nMirrors), eyez),
			vec3(cosf(2 * M_PI*(i + 1) / nMirrors), sinf(2 * M_PI*(i + 1) / nMirrors), mirrorEnd),
			vec3(cosf(2 * M_PI*i / nMirrors), sinf(2 * M_PI*i / nMirrors), mirrorEnd)));
		glutPostRedisplay();
	}

	void MirrorColor(bool _isGold) {
		isGold = _isGold;
		glutPostRedisplay();
	}
};

GPUProgram gpuProgram;
Scene scene;

class FullScreenTexturedQuad {
	unsigned int vao;
public:
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);	

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {
		glBindVertexArray(vao);	
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad.Create();

	gpuProgram.Create(vertexSource, fragmentSource, "fragmentColor");
	gpuProgram.Use();
}

void onDisplay() {
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT );
	scene.SetUniform(gpuProgram.getId());
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();
}


void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'a')scene.IncrementMirrors();
	if (key == 'g')scene.MirrorColor(true);
	if (key == 's')scene.MirrorColor(false);

}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void IdleFunc() {
	static float tend = 0.0f;
	const float dt = 0.01f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(Dt);
	}
	glutPostRedisplay();
}

void onIdle() {
	IdleFunc();
}