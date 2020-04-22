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

const int tessellationLevel = 30;

float rnd() { return (float)rand() / RAND_MAX; }

struct Material {
	vec3 kd, ks, ka;
	float shininess;

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.kd", name);
		kd.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ks", name);
		ks.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ka", name);
		ka.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.shininess", name);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform shininess cannot be set\n");
	}
};

struct Light {
	vec3 La, Le;
	vec4 wLightPos;

	vec4 qmul(vec4 q1, vec4 q2) {
		vec3 d1(q1.x, q1.y, q1.z), d2(q2.x, q2.y, q2.z);
		vec3 ret = d2 * q1.w + d1 * q2.w + cross(d1, d2);
		return vec4(ret.x, ret.y, ret.z, q1.w*q2.w - dot(d1, d2));
	}

	void Animate(float tstart, float tend) {
		float t = tend - tstart;
		vec4 q(sinf(t / 4) * cosf(t) / 2, sinf(t / 4) * sinf(t) / 2, sinf(t / 4) * sqrt(0.75), cosf(t / 4));
		vec4 qinv(-q.x, -q.y, -q.z, q.w);
		wLightPos = qmul( qmul( q, wLightPos), qinv);
	}

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.La", name);
		La.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.Le", name);
		Le.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.wLightPos", name);
		wLightPos.SetUniform(shaderProg, buffer);
	}
};

struct RenderState {
	mat4	           MVP, M, Minv, V, P;
	Material *         material;
	Light			   light;
	Texture *          texture;
	vec3	           wEye;
};

class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;
};

class SimmetricalShader : public Shader {
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv;
		uniform Light light;
		uniform vec3  wEye;

		layout(location = 0) in vec3  vtxPos;
		layout(location = 1) in vec3  vtxNorm;
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight;
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;
			wLight = light.wLightPos.xyz * wPos.w - wPos.xyz * light.wLightPos.w;
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

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
		uniform Light light; 
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;
		in  vec3 wView;
		in  vec3 wLight;
		in  vec2 texcoord;
		
        out vec4 fragmentColor; 

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			vec3 L = normalize(wLight);
			vec3 H = normalize(L + V);
			float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
			radiance += ka * light.La + (kd * cost + material.ks * pow(cosd, material.shininess) * cost /(L+V)/(L+V)) * light.Le;
			
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	SimmetricalShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId());
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.material->SetUniform(getId(), "material");
		state.light.SetUniform(getId(), "light");
		state.texture->SetUniform(getId(), "diffuseTexture");
	}
};

class NPRShader : public Shader {
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv;
		uniform	vec4  wLightPos;
		uniform vec3  wEye; 

		layout(location = 0) in vec3  vtxPos; 
		layout(location = 1) in vec3  vtxNorm;
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal, wView, wLight;
		out vec2 texcoord;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP;
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		uniform sampler2D diffuseTexture;

		in  vec3 wNormal, wView, wLight;
		in  vec2 texcoord;
		out vec4 fragmentColor;

		void main() {
		   vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
		   float y = (dot(N, L) > 0.5) ? 1 : 0.5;
		   if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
		   else						 fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
		}
	)";
public:
	NPRShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId());
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.light.wLightPos.SetUniform(getId(), "wLightPos");

		state.texture->SetUniform(getId(), "diffuseTexture");
	}
};

struct VertexData {
	vec3 position, normal, dU, dV;
	vec2 texcoord;
};

class Geometry {
protected:
	unsigned int vao; 
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	virtual void Draw() = 0;
};

class ParamSurface : public Geometry {
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() {
		nVtxPerStrip = nStrips = 0;
	}
	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N = tessellationLevel, int M = tessellationLevel) {
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));

			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);  
		glEnableVertexAttribArray(1); 
		glEnableVertexAttribArray(2); 

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	void Draw() {
		glBindVertexArray(vao);
		for (int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
};

struct CliffordPlus {
	float f, du, dv;
	CliffordPlus(float f0 = 0, float du0 = 0, float dv0 = 0) { f = f0, du = du0; dv = dv0; }
	CliffordPlus operator+(CliffordPlus r) { return CliffordPlus(f + r.f, du + r.du, dv + r.dv); }
	CliffordPlus operator-(CliffordPlus r) { return CliffordPlus(f - r.f, du - r.du, dv - r.dv); }
	CliffordPlus operator*(CliffordPlus r) { return CliffordPlus(f * r.f, f * r.du + du * r.f, f * r.dv + dv * r.f); }
	CliffordPlus operator/(CliffordPlus r) {
		float l = r.f * r.f;
		return (*this) * CliffordPlus(r.f / l, -r.du / l, -r.dv / l);
	}
};

CliffordPlus TU(float t) { return CliffordPlus(t, 1, 0); }
CliffordPlus TV(float t) { return CliffordPlus(t, 0, 1); }
CliffordPlus Sin(CliffordPlus g) { return CliffordPlus(sin(g.f), cos(g.f) * g.du, cos(g.f) * g.dv); }
CliffordPlus Cos(CliffordPlus g) { return CliffordPlus(cos(g.f), -sin(g.f) * g.du, -sin(g.f) * g.dv); }
CliffordPlus Tan(CliffordPlus g) { return Sin(g) / Cos(g); }
CliffordPlus Log(CliffordPlus g) { return CliffordPlus(logf(g.f), 1 / g.f * g.du, 1 / g.f * g.dv); }
CliffordPlus Exp(CliffordPlus g) { return CliffordPlus(expf(g.f), expf(g.f) * g.du, expf(g.f) * g.dv); }
CliffordPlus Pow(CliffordPlus g, float n) { return CliffordPlus(powf(g.f, n), n * powf(g.f, n - 1) * g.du, n * powf(g.f, n - 1) * g.dv); }


class Dini : public ParamSurface {
	float a, b;
public:
	float minz = 0;

	Dini() { a = 1; b = 0.15; Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = u * 4 * M_PI, V = v * 0.99 + 0.01;

		CliffordPlus x = Cos(TU(U)) * Sin(TV(V)) * a;
		CliffordPlus y = Sin(TU(U)) * Sin(TV(V)) * a;
		CliffordPlus z = (Cos(TV(V)) + Log(Tan(TV(V) / 2))) * a + TU(U) * b;

		if (minz > z.f)minz = z.f;

		vd.position = vec3(x.f, y.f, z.f);
		vec3 drdU(x.du, y.du, z.du);
		vec3 drdV(x.dv, y.dv, z.dv);
		vd.normal = cross(drdU, drdV);
		vd.texcoord = vec2(u, v);

		return vd;
	}
};

class Klein : public ParamSurface {
public:
	Klein() { Create(); }

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = u * 2 * (float)M_PI, V = v * 2 * (float)M_PI;
		CliffordPlus a = Cos(TU(U)) * (Sin(TU(U)) + 1) * 6;
		CliffordPlus b = Sin(TU(U)) * 16;
		CliffordPlus c = (CliffordPlus(1) - Cos(TU(U)) / 2) * 4;
		CliffordPlus x, y, z;
		if ((float)M_PI < U && U <= 2 * (float)M_PI) {
			x = a + c * Cos(TV(V) + (float)M_PI);
			y = b;
			z = c * Sin(TV(V));
		}
		else {
			x = a + c * Cos(TU(U)) * Cos(TV(V));
			y = b + c * Sin(TU(U)) * Cos(TV(V));
			z = c * Sin(TV(V));
		}
		vd.position = vec3(x.f, y.f, z.f);
		vd.dU = vec3(x.du, y.du, z.du);
		vd.dV = vec3(x.dv, y.dv, z.dv);
		vd.normal = cross(vd.dU, vd.dV);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

class Ellipsoid : public ParamSurface {
	vec3 scale = vec3(0.9, 0.7, 0.5);

public:
	Ellipsoid() { Create(); }
	VertexData GenVertexData(float u, float v) {
		float U = u * 2.0f * M_PI, V = v * M_PI;

		VertexData vd;

		CliffordPlus x = Cos(TU(U)) * Sin(TV(V)) * scale.x;
		CliffordPlus y = Sin(TU(U)) * Sin(TV(V)) * scale.y;
		CliffordPlus z = Cos(TV(V)) * scale.z;

		if (z.f < 0) { z = 0; }

		vd.position = vec3(x.f, y.f, z.f);
		vec3 drdU(x.du, y.du, z.du);
		vec3 drdV(x.dv, y.dv, z.dv);
		vd.normal = cross(drdV, drdU);
		vd.texcoord = vec2(u, v);

		return vd;
	}
};

struct Camera { 
	vec3 wEye, wLookat, wVup; 
	float fov, asp, fp, bp;
	bool close=false;
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 130.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 30;
	}
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}
	mat4 P() {
		return mat4(1 / (tan(fov / 2)*asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp*bp / (bp - fp), 0);
	}
	void Animate(float u, float v, mat4 lbTransform) {
		Klein * klein = new Klein();
		VertexData location = klein->GenVertexData(u, v);
		vec3 lbI(lbTransform.m[0][0], lbTransform.m[0][1], lbTransform.m[0][2]);
		wLookat = location.position;
		if(!close)
			wEye = location.position + normalize(location.normal) * 5 - normalize(lbI);
		else
			wEye = location.position + normalize(location.normal) * 2 - normalize(lbI);
		float cosa = dot(normalize(location.normal), normalize(wLookat - wEye));
		float a = acosf(cosa);
		float b = a - (float)M_PI / 2;
		vec3 rotaxis = -cross(wLookat - wEye, location.normal);
		mat4 rot = RotationMatrix(b, rotaxis);
		vec4  Vup= vec4(location.normal.x, location.normal.y, location.normal.z,0)*rot;
		wVup = normalize(vec3(Vup.x, Vup.y, Vup.z));
		delete klein;
	}

	void CloseUp() { close = !close; }
};

struct PlanetTexture : public Texture {
	PlanetTexture() : Texture() {
		int size = 1000;
		glBindTexture(GL_TEXTURE_2D, textureId); 
		std::vector<vec3> image(size * size);
		const vec3 green(0.1, 1, 0.1), blue(0, 0.3, 1);

		for (int x = 0; x < size; x++) for (int y = 0; y < size; y++) {
			if (rnd() < 0.99f) {
				image[y * size + x] = blue;
				if (y != 0 && x != 0 && (image[(y - 1) * size + x].x == green.x && image[y * size + x - 1].x == green.x) && rnd() < 0.95)
					image[y * size + x] = green;
				else if(y!=0 && x!=0 && (image[(y-1) * size + x].x==green.x || image[y * size + x-1].x == green.x) && rnd() < 0.5) 
					image[y * size + x] = green;
			}
			else {
				image[y * size + x] = green;
				if(y!=size-1)image[(y+1) * size + x] = green;
				if (x != size - 1)image[y * size + x+1] = green;
			}
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0, GL_RGB, GL_FLOAT, &image[0]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

struct TreeTexture : public Texture {
	TreeTexture() : Texture() {
		int size = 10;
		glBindTexture(GL_TEXTURE_2D, textureId); 
		std::vector<vec3> image(size * size);
		const vec3 brown(0.5, 0.3, 0), green(0.1, 1, 0);
		for (int x = 0; x < size; x++) for (int y = 0; y < size; y++) {
			image[y * size + x] = (y>size/2) ? green : brown;
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0, GL_RGB, GL_FLOAT, &image[0]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

struct DottedTexture : public Texture {
	DottedTexture() : Texture() {
		int size = 100;
		glBindTexture(GL_TEXTURE_2D, textureId);
		std::vector<vec3> image(size * size);
		const vec3 red(1, 0, 0), black(0, 0, 0);

		float u, v;
		Ellipsoid lb;

		std::vector<vec3> dotCenters;
		dotCenters.push_back(lb.GenVertexData(0, 2.0f / 12.0f).position);
		dotCenters.push_back(lb.GenVertexData(0.2, 3.0f / 12.0f).position);
		dotCenters.push_back(lb.GenVertexData(-0.2, 3.0f / 12.0f).position);
		dotCenters.push_back(lb.GenVertexData(0.3, 1.0f / 12.0f).position);
		dotCenters.push_back(lb.GenVertexData(-0.3, 1.0f / 12.0f).position);
		dotCenters.push_back(lb.GenVertexData(0.35, 4.0f / 12.0f).position);
		dotCenters.push_back(lb.GenVertexData(-0.35, 4.0f / 12.0f).position);

		for (int x = 0; x < size; x++) {
			for (int y = 0; y < size; y++) {

				u = (float)x / (float)size; v = (float)y / (float)size;
				vec3 pos = lb.GenVertexData(u, v).position;
				image[y * size + x] = red;
				for (vec3 center : dotCenters) {
					if (length(pos - center) < 0.16) {
						image[y * size + x] = black;
					}
				}
			}
		}

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0, GL_RGB, GL_FLOAT, &image[0]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

struct Object {
	Shader * shader;
	Material * material;
	Texture * texture;
	ParamSurface * geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
	std::vector<Object *> children;
public:
	Object(Shader * _shader, Material * _material, Texture * _texture, ParamSurface * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	virtual void Draw(RenderState state) {
		state.M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation) * state.M;
		state.Minv = state.Minv * TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
		for (Object * child : children) child->Draw(state);

	}

	void AddChild(Object * child, float posU, float posV, vec3 childToZero, vec3 childDir) {
		vec3 posOnPlanet = geometry->GenVertexData(posU, posV).position;
		vec3 normalOnPlanet = geometry->GenVertexData(posU, posV).normal;
		float cosa = dot(normalize(normalOnPlanet), normalize(childDir));
		child->translation = posOnPlanet + normalize(normalOnPlanet) * length(childToZero);
		child->rotationAxis = -cross(normalOnPlanet, childDir);
		child->rotationAngle = acosf(cosa);
		child->scale = vec3(1, 1, 1);
		children.push_back(child);
	}

};

struct LadyBugObject : public Object {
	ParamSurface * klein = new Klein();
	float angle=0;
	float V = 1;
	mat4  invtransform;
public:
	mat4 transform;
	float u = 0.5, v = 0.2;

	LadyBugObject(Shader * _shader, Material * _material, Texture * _texture, ParamSurface * _geometry) :
		Object(_shader, _material, _texture, _geometry) {
	}

	void Animate(float tstart, float tend) {
		float dt = tend - tstart;

		VertexData location = klein->GenVertexData(u, v);

		vec3 originalPos = location.position;
		float du = V * dt * cosf(angle) / length(location.dU);
		float dv = V * dt * sinf(angle) / length(location.dV);
		u += du; v += dv;

		location = klein->GenVertexData(u, v);

		if (u > 1)u = 0;
		if (u < 0)u = 1;

		vec3 r = location.position;
		vec3 i = normalize(r-originalPos);
		vec3 k = normalize(location.normal);
		vec3 j = cross(k, i);

		transform = mat4(i.x,i.y,i.z,0.0f,
						j.x,j.y,j.z,0.0f,
						k.x,k.y,k.z,0.0f,
						r.x,r.y,r.z,1.0f);

		invtransform = mat4(i.x, j.x, k.x, 0.0f,
							i.y, j.y, k.y, 0.0f,
							i.z, j.z, k.z, 0.0f,
							-r.x, -r.y, -r.z, 1.0f);
	}

	void Draw(RenderState state) {
		state.M = transform * state.M;
		state.Minv = state.Minv * invtransform;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	void IncAngle() { angle+=45.0f/180.0f*(float)M_PI; }
	void DecAngle() { angle-= 45.0f / 180.0f*(float)M_PI; }
	void print() { printf("%f %f", u, v); }
};

class Scene {
	Object * kleinObject;
	LadyBugObject * ladybugObject;
	Camera camera;
	Light light;
public:
	void Build() {
		Shader * simmetricalShader = new SimmetricalShader();
		Shader * nprshader = new NPRShader();

		Material * material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		Texture * planet = new PlanetTexture();
		Texture * lbTexture = new DottedTexture();
		Texture * diniTexture = new TreeTexture();

		Dini * dini = new Dini();
		Klein * klein = new Klein();
		Ellipsoid * ladybug = new Ellipsoid();

		kleinObject = new Object(simmetricalShader, material0, planet, klein);

		vec3 diniToZero = (vec3(0, 0, 0) - vec3(0, 0, dini->minz));
		
		Object * diniObj0 = new Object(simmetricalShader, material0, diniTexture, dini);
		Object * diniObj1 = new Object(simmetricalShader, material0, diniTexture, dini);
		Object * diniObj2 = new Object(simmetricalShader, material0, diniTexture, dini);
		Object * diniObj3 = new Object(simmetricalShader, material0, diniTexture, dini);
		Object * diniObj4 = new Object(simmetricalShader, material0, diniTexture, dini);
		
		kleinObject->AddChild(diniObj0, 0.4, 0.4, diniToZero, diniToZero);
		kleinObject->AddChild(diniObj1, 0.1, 0.1, diniToZero, diniToZero);
		kleinObject->AddChild(diniObj2, 0.2, 0.7, diniToZero, diniToZero);
		kleinObject->AddChild(diniObj3, 0.7, 0.1, diniToZero, diniToZero);
		kleinObject->AddChild(diniObj4, 0.7, 0.5, diniToZero, diniToZero);

		ladybugObject = new LadyBugObject(nprshader, material0, lbTexture, ladybug);

		kleinObject->AddChild(ladybugObject, ladybugObject->u, ladybugObject->v, vec3(0,0,0), vec3(0,0,1));

		VertexData location = klein->GenVertexData(ladybugObject->u, ladybugObject->v);
		vec3 lbI(ladybugObject->transform.m[0][0], ladybugObject->transform.m[0][1], ladybugObject->transform.m[0][2]);
		camera.wLookat = location.position;
		camera.wEye = location.position + normalize(location.normal) * 5 - normalize(lbI);
		float cosa = dot(normalize(location.normal), normalize(camera.wLookat - camera.wEye));
		float a = acosf(cosa);
		float b = a - (float)M_PI / 2;
		vec3 rotaxis = -cross(camera.wLookat - camera.wEye, location.normal);
		mat4 rot = RotationMatrix(b, rotaxis);
		vec4  Vup = vec4(location.normal.x, location.normal.y, location.normal.z, 0)*rot;
		camera.wVup = normalize(vec3(Vup.x, Vup.y, Vup.z));

		light.wLightPos = vec4(0, 0, 1, 0);
		light.La = vec3(1, 1, 1);
		light.Le = vec3(3, 3, 3);


	}
	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.light = light;
		state.M = mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
		state.Minv = mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
		kleinObject->Draw(state);
	}

	void Animate(float tstart, float tend) {
		light.Animate(tstart, tend); 
		ladybugObject->Animate(tstart, tend);
		camera.Animate(ladybugObject->u,ladybugObject->v, ladybugObject->transform);
	}

	void OnA() { ladybugObject->IncAngle(); }
	void OnS() { ladybugObject->DecAngle(); }
	void OnSpace() { camera.CloseUp(); }
};

Scene scene;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) { 
	if (key == 'a') scene.OnA();
	if (key == 's') scene.OnS();
	if (key == ' ') scene.OnSpace();
}

void onKeyboardUp(unsigned char key, int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) { }

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	static float tend = 0;
	const float dt = 0.1;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}