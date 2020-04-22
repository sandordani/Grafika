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
// Nev    : Sándor Dániel
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

const char * const vertexSource = R"(
	#version 330
	precision highp float;

	uniform mat4 MVP;
	layout(location = 0) in vec2 vp;
	layout(location = 1) in vec2 vtxUV;

	out vec2 texCoord;

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;
		texCoord=vtxUV;
	}
)";

const char * const fragmentSource = R"(
	#version 330
	precision highp float;
	
	uniform vec3 color;
	uniform int textured;
	uniform sampler2D textureUnit;

	in vec2 texCoord;
	out vec4 outColor;

	void main() {
		if(textured==1)
			outColor = texture(textureUnit, texCoord);
		else
			outColor = vec4(color, 1);
	}
)";



GPUProgram gpuProgram;

struct Camera {
	float wCx, wCy;
	float wWx, wWy;
	bool follow;
public:
	Camera() {
		Animate(0,0);
	}

	mat4 V() {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, 0, 1);
	}

	mat4 P() {
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, 0, 1);
	}

	mat4 Pinv() {
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	void Animate(float x, float y) {
		if (follow) {
			wCx =x;
			wCy = y;
			wWx = 10;
			wWy = 10;
		}
		else {
			wCx = 0;
			wCy = 0;
			wWx = 57.5f;
			wWy = 57.5f;
		}
		
	}
};

Camera camera;

struct Spline {

	unsigned int vao, vbo;
	std::vector<vec2> CPs;
	float tension;
	unsigned int nTesselatedVertices;

	float Hermite(float p0, float v0, float t0,
		float p1, float v1, float t1,
		float t) {
		float a3, a2, a1, a0;
		a0 = p0;
		a1 = v0;
		a2 = 3 * (p1 - p0) / (t1 - t0) / (t1 - t0) - (v1 + 2 * v0) / (t1 - t0);
		a3 = 2 * (p0 - p1) / (t1 - t0) / (t1 - t0) / (t1 - t0) + (v1 + v0) / (t1 - t0) / (t1 - t0);
		float r = a3 * ((t - t0)*(t - t0)*(t - t0)) + a2 * ((t - t0)*(t - t0)) + a1 * (t - t0) + a0;
		return r;
	}

	float dHermite(float p0, float v0, float t0,
		float p1, float v1, float t1,
		float t) {
		float a3, a2, a1, a0;
		a0 = p0;
		a1 = v0;
		a2 = 3 * (p1 - p0) / (t1 - t0) / (t1 - t0) - (v1 + 2 * v0) / (t1 - t0);
		a3 = 2 * (p0 - p1) / (t1 - t0) / (t1 - t0) / (t1 - t0) + (v1 + v0) / (t1 - t0) / (t1 - t0);
		float r = 3 * a3 * ((t - t0)*(t - t0)) + 2 * a2 * (t - t0) + a1;
		return r;
	}

	virtual float tStart() { return CPs[0].x; }
	virtual float tEnd() { return CPs[CPs.size()-1].x; }

public:
	Spline(float tens, unsigned int tesselat) {

		tension = tens;
		nTesselatedVertices = tesselat;

		vec4 first = vec4(-1.5, 0, 0, 1) * camera.Pinv() * camera.Vinv();
		vec4 second = vec4(-1, 0, 0, 1) * camera.Pinv() * camera.Vinv();
		vec4 beforelast = vec4(1, 0, 0, 1) * camera.Pinv() * camera.Vinv();
		vec4 last = vec4(1.5, 0, 0, 1) * camera.Pinv() * camera.Vinv();
		
		CPs.push_back(vec2(first.x, first.y));
		CPs.push_back(vec2(second.x, second.y));
		CPs.push_back(vec2(beforelast.x, beforelast.y));
		CPs.push_back(vec2(last.x, last.y));


	}

	void Create() {
		glGenVertexArrays(1, &vao);	
		glBindVertexArray(vao);	
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glEnableVertexAttribArray(0); 
		glVertexAttribPointer(0, 
			2, GL_FLOAT, GL_FALSE,
			0, NULL);
	}

	void AddControlPoint(float cX, float cY) {
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();

		for (std::vector<vec2>::iterator it = CPs.begin(); it != CPs.end(); ++it) {
			if (it->x == wVertex.x) break;
			if (it->x > wVertex.x) {
				CPs.insert(it, vec2(wVertex.x, wVertex.y));
				break;
			}
		}
		
		
	}

	float r(float x) {
		for (unsigned int i = 0; i < CPs.size()-1; i++) {
			if (CPs[i].x <= x && CPs[i + 1].x >= x) {
				float v0, v1;

				if (i == 0) {
					v0 = 0;
				}
				else v0 = ((CPs[i + 1].y - CPs[i].y) / (CPs[i + 1].x - CPs[i].x) + (CPs[i].y - CPs[i - 1].y) / (CPs[i].x - CPs[i - 1].x))*(1 - tension);

				if (i + 1 == (CPs.size() - 1)) {
					v1 = 0;
				}
				else v1 = ((CPs[i + 2].y - CPs[i + 1].y) / (CPs[i + 2].x - CPs[i + 1].x) + (CPs[i + 1].y - CPs[i].y) / (CPs[i + 1].x - CPs[i].x))*(1 - tension);

				return Hermite(CPs[i].y, v0, CPs[i].x, CPs[i + 1].y, v1, CPs[i + 1].x, x);
			}
		}
		return 0;
	}

	float Derivative(float x) {
		for (unsigned int i = 0; i < CPs.size() - 1; i++) {
			if (CPs[i].x <= x && CPs[i + 1].x >= x) {
				float v0, v1;

				if (i == 0) {
					v0 = 0;
				}
				else v0 = ((CPs[i + 1].y - CPs[i].y) / (CPs[i + 1].x - CPs[i].x) + (CPs[i].y - CPs[i - 1].y) / (CPs[i].x - CPs[i - 1].x))*(1 - tension);

				if (i + 1 == (CPs.size() - 1)) {
					v1 = 0;
				}
				else v1 = ((CPs[i + 2].y - CPs[i + 1].y) / (CPs[i + 2].x - CPs[i + 1].x) + (CPs[i + 1].y - CPs[i].y) / (CPs[i + 1].x - CPs[i].x))*(1 - tension);

				return dHermite(CPs[i].y, v0, CPs[i].x, CPs[i + 1].y, v1, CPs[i + 1].x, x);
			}
		}
		return 0;
	}

	void Draw(vec3 color) {

			mat4 VPTransform = camera.V() * camera.P();
			VPTransform.SetUniform(gpuProgram.getId(), "MVP");

			int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(colorLocation, color.x,color.y,color.z);

			std::vector<float> vertices;
			for (unsigned int i = 0; i < nTesselatedVertices; i++) {
				float x = tStart() + (tEnd() - tStart())*i / (nTesselatedVertices - 1);
				float y = r(x);
				vertices.push_back(x);
				vertices.push_back(y);
				
				vertices.push_back(x);
				vertices.push_back(-28.75f);

				
			}

			glBindVertexArray(vao);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, 
				sizeof(float)*vertices.size(), 
				&vertices[0],
				GL_DYNAMIC_DRAW);

			

			glDrawArrays(GL_TRIANGLE_STRIP, 0, nTesselatedVertices*2);
		
	}
};

Spline* mountain;
Spline* road;

struct Bike {

	unsigned int vaoWheel, vboWheel, vaoMan, vboMan, vaoLeg, vboLeg;
	vec2 wTranslate, n, foot1, foot2;
	std::vector<float> wheelVertices, manVertices, legVertices;
	float r, v, alpha, uleg, lleg, F, mg, ro;
	bool right;

	vec2 Knee(vec2 foot) {

		vec2 c = vec2(0, 1.1f)-foot;
		float lc = length(c);

		float cosa = (lc*lc + uleg*uleg - lleg*lleg) / (2 * uleg*lc);
		float angle = acosf(cosa);

		angle += (float)M_PI;

		vec2 ret = normalize(c)*uleg ;

		if(right)
			ret = vec2(ret.x*cosf(angle)-ret.y*sinf(angle), ret.x*sinf(angle)+ret.y*cosf(angle))+  vec2(0, 1.1f);
		else
			ret = vec2(ret.x*cosf(angle) + ret.y*sinf(angle), -ret.x*sinf(angle) + ret.y*cosf(angle)) + vec2(0, 1.1f);
		
		return ret;
	}
public:
	Bike() {
		r = 1.0f;
		alpha = 0.0f;
		F = 1840.0f;
		mg = 1840.0f;
		ro = 160.0f;

		foot1.x = 0.5f;
		foot1.y = 0.0f;
		foot2.x = -0.5f;
		foot2.y = 0.0f;

		uleg = 1.0f;
		lleg = 1.2f;


		for (unsigned int i = 0; i < 100; i++) {

			float x = r * sinf(2 * (float)M_PI*i / 99.0f);
			float y = r * cosf(2 * (float)M_PI*i / 99.0f);

			wheelVertices.push_back(x);
			wheelVertices.push_back(y);

		}

		for (unsigned int i = 0; i < 10; i++)
		{
			wheelVertices.push_back(0.0f);
			wheelVertices.push_back(0.0f);

			wheelVertices.push_back(r * sinf(2.0f*(float)M_PI*i/9.0f));
			wheelVertices.push_back(r * cosf(2.0f*(float)M_PI*i/9.0f));
		}
		

		manVertices.push_back(0.0f);
		manVertices.push_back(0.0f);

		manVertices.push_back(0.0f);
		manVertices.push_back(1.1f);

		manVertices.push_back(-0.3f);
		manVertices.push_back(1.1f);

		manVertices.push_back(0.3f);
		manVertices.push_back(1.1f);

		
		manVertices.push_back(0.0f);
		manVertices.push_back(1.1f);

		
		manVertices.push_back(0.0f);
		manVertices.push_back(2.0f);

		manVertices.push_back(-0.5f);
		manVertices.push_back(2.0f);

		manVertices.push_back(-0.9f);
		manVertices.push_back(2.3f);

		manVertices.push_back(-0.5f);
		manVertices.push_back(2.0f);

		manVertices.push_back(0.5f);
		manVertices.push_back(2.0f);

		manVertices.push_back(0.9f);
		manVertices.push_back(2.3f);

		manVertices.push_back(0.5f);
		manVertices.push_back(2.0f);

		manVertices.push_back(0.0f);
		manVertices.push_back(2.0f);

		
		manVertices.push_back(0.0f);
		manVertices.push_back(2.5f);

		float rHead = 0.5;

		for (unsigned int i = 0; i < 100; i++) {

			float x = rHead * sinf(2 * (float)M_PI * i / 99.0f - (float)M_PI);
			float y = rHead * cosf(2 * (float)M_PI * i / 99.0f - (float)M_PI) + 3;

			manVertices.push_back(x);
			manVertices.push_back(y);
		}


		legVertices.push_back(foot1.x);
		legVertices.push_back(foot1.y);

		vec2 knee1 = Knee(foot1);
		legVertices.push_back(knee1.x);
		legVertices.push_back(knee1.y);

		legVertices.push_back(0.0f);
		legVertices.push_back(1.1f);

		vec2 knee2 = Knee(foot2);
		legVertices.push_back(knee2.x);
		legVertices.push_back(knee2.y);

		legVertices.push_back(foot2.x);
		legVertices.push_back(foot2.y);

		wTranslate.x = -10.0f;
		wTranslate.y = 1.0f;


	}

	void Create() {
		glGenVertexArrays(1, &vaoWheel);	
		glBindVertexArray(vaoWheel);		
		glGenBuffers(1, &vboWheel);	
		glBindBuffer(GL_ARRAY_BUFFER, vboWheel);

		glEnableVertexAttribArray(0);  
		glVertexAttribPointer(0,     
			2, GL_FLOAT, GL_FALSE, 
			0, NULL);

		glGenVertexArrays(1, &vaoMan);
		glBindVertexArray(vaoMan);	
		glGenBuffers(1, &vboMan);
		glBindBuffer(GL_ARRAY_BUFFER, vboMan);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,
			2, GL_FLOAT, GL_FALSE,
			0, NULL);

		glGenVertexArrays(1, &vaoLeg);
		glBindVertexArray(vaoLeg);
		glGenBuffers(1, &vboLeg);
		glBindBuffer(GL_ARRAY_BUFFER, vboLeg);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,
			2, GL_FLOAT, GL_FALSE,
			0, NULL);
	}

	void Draw() {

		mat4 MVPTransform = MWheel() * camera.V() * camera.P();
		MVPTransform.SetUniform(gpuProgram.getId(), "MVP");

		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(colorLocation, 0.5, 0, 1);

		glBindVertexArray(vaoWheel);
		glBindBuffer(GL_ARRAY_BUFFER, vboWheel);
		glBufferData(GL_ARRAY_BUFFER, 
			sizeof(float)*wheelVertices.size(),
			&wheelVertices[0],
			GL_STATIC_DRAW);

		glDrawArrays(GL_LINE_STRIP, 0, wheelVertices.size()/2);

		MVPTransform = MMan() * camera.V() * camera.P();
		MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
		
		glBindVertexArray(vaoMan);
		glBindBuffer(GL_ARRAY_BUFFER, vboMan);
		glBufferData(GL_ARRAY_BUFFER, 
			sizeof(float)*manVertices.size(),
			&manVertices[0],
			GL_STATIC_DRAW);

		glDrawArrays(GL_LINE_STRIP, 0, manVertices.size()/2);

		
		glBindVertexArray(vaoLeg);
		glBindBuffer(GL_ARRAY_BUFFER, vboLeg);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(float)*legVertices.size(),
			&legVertices[0],
			GL_DYNAMIC_DRAW);

		glDrawArrays(GL_LINE_STRIP, 0, legVertices.size() / 2);

	}

	void Animate(float t) {
		wTranslate = wTranslate - n * r;

		float oldDy = road->Derivative(wTranslate.x);

		float sina = oldDy / sqrt(1+oldDy*oldDy);
		if (!right)sina *= -1;

		v = (F - mg * sina) / ro;
		if (!right)v *= -1;

		float s = v * t;
		float dx = s/sqrt(1+ oldDy * oldDy);

		wTranslate.x += dx;

		if (wTranslate.x >= 28.75f - r) {
			right = false;
			wTranslate.x = 28.75f - r;
		}

		if (wTranslate.x <= -28.75f + r) {
			right = true;
			wTranslate.x = -28.75f + r;
		}
		wTranslate.y = road->r(wTranslate.x);

		float dy = road->Derivative(wTranslate.x);
		vec2 i = vec2(1 / sqrt(1 + dy * dy), dy / sqrt(1 + dy * dy));
		n = vec2(-i.y, i.x);

		wTranslate = wTranslate + n * r;

		alpha += s/r;

		legVertices[0] = foot1.x = 0.5f*sinf(alpha);
		legVertices[1] = foot1.y = 0.5f*cosf(alpha);
		legVertices[8] = foot2.x = -0.5f * sinf(alpha);
		legVertices[9] = foot2.y = -0.5f*cosf(alpha);

		vec2 knee1 = Knee(foot1);
		legVertices[2] = knee1.x;
		legVertices[3] = knee1.y;

		vec2 knee2 = Knee(foot2);
		legVertices[6] = knee2.x;
		legVertices[7] = knee2.y;


	}

	mat4 MMan() {
		mat4 Mtranslate(1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 0, 0,
						wTranslate.x, wTranslate.y, 0, 1); 

		return  Mtranslate;	
	}

	mat4 MWheel() {
		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTranslate.x, wTranslate.y, 0, 1);

		mat4 Mrotate(sinf(alpha), cosf(alpha), 0, 0,
			-cosf(alpha), sinf(alpha), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		return  Mrotate * Mtranslate;
	}

};

Bike* bike;

struct Background {
	unsigned int textureId, vao, vbo[2];
	std::vector<float> vtxs, uvs;
	Texture* texture;

	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(2,vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

		vtxs.push_back(-10);
		vtxs.push_back(-10);
		
		vtxs.push_back(10);
		vtxs.push_back(-10);

		vtxs.push_back(10);
		vtxs.push_back(10);

		vtxs.push_back(-10);
		vtxs.push_back(10);

		glBufferData(GL_ARRAY_BUFFER, vtxs.size() * sizeof(float), &vtxs[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);

		uvs.push_back(0);
		uvs.push_back(0);

		uvs.push_back(1);
		uvs.push_back(0);

		uvs.push_back(1);
		uvs.push_back(1);

		uvs.push_back(0);
		uvs.push_back(1);

		glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(float), &uvs[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		
		mountain = new Spline(0.5f, 1000);
		mountain->Create();
		mountain->AddControlPoint(0.5f, 0.5f);
		mountain->AddControlPoint(-0.5f, 0.7f);
		mountain->AddControlPoint(0.0f, -0.3f);

		std::vector<vec4> image( windowWidth * windowHeight );
		for (unsigned i = 0; i < windowWidth; i++)
		{
			float cX = 2.0f * i / windowWidth - 1;
			for (unsigned int j =0; j <windowHeight; j++)
			{
				float cY = 1.0f - 2.0f * (windowHeight-j) / windowHeight;
				vec4 wPoint=vec4(cX, cY, 0, 1)*camera.Vinv() * camera.Pinv();
				if (wPoint.y > mountain->r(wPoint.x))
					image[j*windowWidth + i] = vec4(0.0f, 0.7f, 1.0f, 1.0f);
				else
					image[j*windowWidth + i] = vec4(0.5f, 0.5f, 0.8f, 1.0f);
			}
		}


		texture = new Texture(windowWidth, windowHeight, image);
	}

	void Draw() {

		glBindVertexArray(vao);
		
		mat4 VPTransform = mat4(0.1f,0.0f,0.0f,0.0f,
								0.0f,0.1f,0.0f,0.0f,
								0.0f,0.0f,1.0f,0.0f,
								0.0f,0.0f,0.0f,1.0f);
		VPTransform.SetUniform(gpuProgram.getId(), "MVP");

		texture->SetUniform(gpuProgram.getId(), "textureUnit");
		
		glDrawArrays(GL_TRIANGLE_FAN, 0, vtxs.size() / 2);
	}

};

Background background;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	background.Create();

	road = new Spline(-0.5f, 1000);
	road->Create();

	bike = new Bike();
	bike->Create();
	

	glClearColor(0.0f, 0.0f,0.0f, 0.0f);

	gpuProgram.Create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	int texturedLocation = glGetUniformLocation(gpuProgram.getId(),"textured");

	glUniform1i(texturedLocation, 1);
	background.Draw();

	glUniform1i(texturedLocation, 0);
	road->Draw(vec3(1, 0.5, 0));
	bike->Draw();

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();
	if (key == ' ') camera.follow=(!camera.follow);
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
		if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  
		float cX = 2.0f * pX / windowWidth - 1;	
		float cY = 1.0f - 2.0f * pY / windowHeight;
		road->AddControlPoint(cX, cY);
		glutPostRedisplay();
	}
}

void IdleFunc() {
	static float tend = 0.0f;
	const float dt = 0.01f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		bike->Animate(Dt);
		camera.Animate(bike->wTranslate.x,bike->wTranslate.y+1.1f);
	}
	glutPostRedisplay();
}

void onIdle() {
	IdleFunc();
}


