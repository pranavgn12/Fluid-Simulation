#include "raylib.h"
#include <math.h>
#include <string.h>
#include <stdbool.h>

#define N 400
#define SCREEN_W 800
#define SCREEN_H 800
#define CELL ((float)SCREEN_W / N)

#define MOUSE_RADIUS 10
#define DYE_RATE 3.0f
#define MOUSE_FORCE 8.0f
#define PRESSURE_ITERATIONS 250

// Dye fading: higher means faster disappearance.
#define DYE_FADE_RATE 0.01f

static float dt;
static float rho;
static float h;
static int speed;

static float u[N][N];
static float v[N][N];
static float u_tmp[N][N];
static float v_tmp[N][N];

static float pressure[N][N];
static float pressure_tmp[N][N];
static float divergence[N][N];

// Dye concentration and average colour.
static float dye[N][N];
static float dye_r[N][N];
static float dye_g[N][N];
static float dye_b[N][N];

static float dye_tmp[N][N];
static float dye_r_tmp[N][N];
static float dye_g_tmp[N][N];
static float dye_b_tmp[N][N];


// --------------------------------------------------
// Bilinear interpolation
// --------------------------------------------------

static float sample(float field[N][N], float x, float y)
{
    if (x < 0.0f) x = 0.0f;
    if (y < 0.0f) y = 0.0f;

    if (x > N - 1.001f) x = N - 1.001f;
    if (y > N - 1.001f) y = N - 1.001f;

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float a = x - x0;
    float b = y - y0;

    float q00 = field[y0][x0];
    float q10 = field[y0][x1];
    float q01 = field[y1][x0];
    float q11 = field[y1][x1];

    return
    (1.0f - a) * (1.0f - b) * q00 +
    a * (1.0f - b) * q10 +
    (1.0f - a) * b * q01 +
    a * b * q11;
}


// --------------------------------------------------
// Inject velocity and coloured dye
// A new random colour is chosen once per mouse drag.
// --------------------------------------------------

static void inject_mouse(void)
{
    static Vector2 previous = {0};
    static bool was_down = false;
    static Color drag_color = WHITE;

    if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        was_down = false;
        return;
    }

    Vector2 m = GetMousePosition();

    // Choose a colour only when a new drag begins.
    if (!was_down) {
        previous = m;
        was_down = true;

        drag_color = (Color) {
            GetRandomValue(50, 255),
            GetRandomValue(50, 255),
            GetRandomValue(50, 255),
            255
        };
    }

    float dx = m.x - previous.x;
    float dy = m.y - previous.y;

    float distance = sqrtf(dx * dx + dy * dy);

    int steps = (int)ceilf(distance / (CELL * 0.5f));

    if (steps < 1)
        steps = 1;

    float amount = DYE_RATE * dt;
    float force_x = dx * MOUSE_FORCE * dt / steps;
    float force_y = dy * MOUSE_FORCE * dt / steps;

    for (int i = 0; i <= steps; i++) {

        float t = (float)i / steps;

        float px = previous.x + dx * t;
        float py = previous.y + dy * t;

        int cx = (int)(px / CELL);
        int cy = (int)(py / CELL);

        for (int y = cy - MOUSE_RADIUS;
             y <= cy + MOUSE_RADIUS; y++) {

            for (int x = cx - MOUSE_RADIUS;
                 x <= cx + MOUSE_RADIUS; x++) {

                if (x < 1 || x >= N - 1 ||
                    y < 1 || y >= N - 1)
                    continue;

                float fx = (float)x - cx;
            float fy = (float)y - cy;

            if (fx * fx + fy * fy >
                MOUSE_RADIUS * MOUSE_RADIUS)
                continue;

            // Add velocity in the direction of mouse movement.
            u[y][x] += force_x;
            v[y][x] += force_y;

            // Blend the incoming colour with the existing dye.
            float old_d = dye[y][x];
            float new_d = old_d + amount;

            if (new_d > 1.0f)
                new_d = 1.0f;

                if (new_d > 0.0f) {

                    // Use the incoming amount as a mixing weight.
                    // Concentration stays bounded, avoiding whitening.
                    float denominator = old_d + amount;

                    if (denominator > 0.000001f) {

                        float incoming_r = drag_color.r / 255.0f;
                        float incoming_g = drag_color.g / 255.0f;
                        float incoming_b = drag_color.b / 255.0f;

                        dye_r[y][x] =
                        (dye_r[y][x] * old_d +
                        incoming_r * amount) / denominator;

                        dye_g[y][x] =
                        (dye_g[y][x] * old_d +
                        incoming_g * amount) / denominator;

                        dye_b[y][x] =
                        (dye_b[y][x] * old_d +
                        incoming_b * amount) / denominator;
                    }

                    dye[y][x] = new_d;
                }
                 }
             }
    }

    previous = m;
}


// --------------------------------------------------
// Advect velocity using backward tracing
// --------------------------------------------------

static void advect_velocity(void)
{
    memcpy(u_tmp, u, sizeof(u));
    memcpy(v_tmp, v, sizeof(v));

    for (int y = 1; y < N - 1; y++) {
        for (int x = 1; x < N - 1; x++) {

            float xb = x - u[y][x] * dt / h;
            float yb = y - v[y][x] * dt / h;

            u_tmp[y][x] = sample(u, xb, yb);
            v_tmp[y][x] = sample(v, xb, yb);
        }
    }

    memcpy(u, u_tmp, sizeof(u));
    memcpy(v, v_tmp, sizeof(v));
}


// --------------------------------------------------
// Enforce solid-wall normal velocities
// --------------------------------------------------

static void apply_wall_boundaries(void)
{
    for (int y = 0; y < N; y++) {
        u[y][0] = 0.0f;
        u[y][N - 1] = 0.0f;
    }

    for (int x = 0; x < N; x++) {
        v[0][x] = 0.0f;
        v[N - 1][x] = 0.0f;
    }
}


// --------------------------------------------------
// Compute velocity divergence
// --------------------------------------------------

static void compute_divergence(void)
{
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {

            int xl = (x > 0) ? x - 1 : x;
            int yu = (y > 0) ? y - 1 : y;

            divergence[y][x] =
            (u[y][x] - u[y][xl]) / h +
            (v[y][x] - v[yu][x]) / h;
        }
    }
}


// --------------------------------------------------
// Solve pressure using Jacobi iterations
// --------------------------------------------------

static void solve_pressure(void)
{
    memset(pressure, 0, sizeof(pressure));
    memset(pressure_tmp, 0, sizeof(pressure_tmp));

    float scale = rho * h * h / dt;

    for (int iter = 0;
         iter < PRESSURE_ITERATIONS; iter++) {

        for (int y = 1; y < N - 1; y++) {
            for (int x = 1; x < N - 1; x++) {

                pressure_tmp[y][x] =
                (pressure[y][x - 1] +
                pressure[y][x + 1] +
                pressure[y - 1][x] +
                pressure[y + 1][x] -
                scale * divergence[y][x]) * 0.25f;
            }
        }

        // Approximate zero-normal pressure gradient.
        for (int x = 0; x < N; x++) {
            pressure_tmp[0][x] =
            pressure_tmp[1][x];

            pressure_tmp[N - 1][x] =
            pressure_tmp[N - 2][x];
        }

        for (int y = 0; y < N; y++) {
            pressure_tmp[y][0] =
            pressure_tmp[y][1];

            pressure_tmp[y][N - 1] =
            pressure_tmp[y][N - 2];
        }

        memcpy(pressure, pressure_tmp, sizeof(pressure));
         }
}


// --------------------------------------------------
// Correct velocity using the pressure gradient
// --------------------------------------------------

static void correct_velocity(void)
{
    float factor = dt / (rho * h);

    memcpy(u_tmp, u, sizeof(u));
    memcpy(v_tmp, v, sizeof(v));

    for (int y = 1; y < N - 1; y++) {
        for (int x = 1; x < N - 1; x++) {

            float dpdx =
            pressure[y][x + 1] - pressure[y][x];

            float dpdy =
            pressure[y + 1][x] - pressure[y][x];

            u_tmp[y][x] =
            u[y][x] - factor * dpdx;

            v_tmp[y][x] =
            v[y][x] - factor * dpdy;
        }
    }

    memcpy(u, u_tmp, sizeof(u));
    memcpy(v, v_tmp, sizeof(v));
}


// --------------------------------------------------
// Pressure projection
// --------------------------------------------------

static void project_velocity(void)
{
    compute_divergence();
    solve_pressure();
    correct_velocity();
    apply_wall_boundaries();
}


// --------------------------------------------------
// Advect dye and all colour channels together
// Bilinear interpolation transports colour as well.
// --------------------------------------------------

static void advect_dye(void)
{
    memcpy(dye_tmp, dye, sizeof(dye));
    memcpy(dye_r_tmp, dye_r, sizeof(dye_r));
    memcpy(dye_g_tmp, dye_g, sizeof(dye_g));
    memcpy(dye_b_tmp, dye_b, sizeof(dye_b));

    for (int y = 1; y < N - 1; y++) {
        for (int x = 1; x < N - 1; x++) {

            float xb = x - u[y][x] * dt / h;
            float yb = y - v[y][x] * dt / h;

            dye_tmp[y][x] =
            sample(dye, xb, yb);

            dye_r_tmp[y][x] =
            sample(dye_r, xb, yb);

            dye_g_tmp[y][x] =
            sample(dye_g, xb, yb);

            dye_b_tmp[y][x] =
            sample(dye_b, xb, yb);
        }
    }

    memcpy(dye, dye_tmp, sizeof(dye));
    memcpy(dye_r, dye_r_tmp, sizeof(dye_r));
    memcpy(dye_g, dye_g_tmp, sizeof(dye_g));
    memcpy(dye_b, dye_b_tmp, sizeof(dye_b));

    // Exponential fading: frame-rate independent.
    float fade = expf(-DYE_FADE_RATE * dt);

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {

            dye[y][x] *= fade;

            // Remove tiny residuals so old dye eventually disappears.
            if (dye[y][x] < 0.001f) {
                dye[y][x] = 0.0f;
                dye_r[y][x] = 0.0f;
                dye_g[y][x] = 0.0f;
                dye_b[y][x] = 0.0f;
            }
        }
    }
}


// --------------------------------------------------
// Initialize arrays
// --------------------------------------------------

static void init_fluid_arrays(void)
{
    memset(u, 0, sizeof(u));
    memset(v, 0, sizeof(v));

    memset(u_tmp, 0, sizeof(u_tmp));
    memset(v_tmp, 0, sizeof(v_tmp));

    memset(pressure, 0, sizeof(pressure));
    memset(pressure_tmp, 0, sizeof(pressure_tmp));
    memset(divergence, 0, sizeof(divergence));

    memset(dye, 0, sizeof(dye));
    memset(dye_r, 0, sizeof(dye_r));
    memset(dye_g, 0, sizeof(dye_g));
    memset(dye_b, 0, sizeof(dye_b));

    memset(dye_tmp, 0, sizeof(dye_tmp));
    memset(dye_r_tmp, 0, sizeof(dye_r_tmp));
    memset(dye_g_tmp, 0, sizeof(dye_g_tmp));
    memset(dye_b_tmp, 0, sizeof(dye_b_tmp));
}


// --------------------------------------------------
// Render coloured dye
// Concentration controls opacity, not RGB brightness.
// --------------------------------------------------

static void render_fluid(void)
{
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {

            float d = dye[y][x];

            if (d < 0.01f)
                continue;

            if (d > 1.0f)
                d = 1.0f;

            int r = (int)(255.0f * dye_r[y][x]);
            int g = (int)(255.0f * dye_g[y][x]);
            int b = (int)(255.0f * dye_b[y][x]);

            if (r < 0) r = 0;
            if (r > 255) r = 255;

            if (g < 0) g = 0;
            if (g > 255) g = 255;

            if (b < 0) b = 0;
            if (b > 255) b = 255;

            unsigned char alpha =
            (unsigned char)(255.0f * d);

            DrawRectangle(
                (int)(x * CELL),
                          (int)(y * CELL),
                          (int)ceilf(CELL),
                          (int)ceilf(CELL),
                          (Color){
                              (unsigned char)r,
                          (unsigned char)g,
                          (unsigned char)b,
                          alpha
                          }
            );
        }
    }
}


// --------------------------------------------------
// Main
// --------------------------------------------------

int main(void)
{
    InitWindow(
        SCREEN_W,
        SCREEN_H,
        "Coloured Fluid Simulation"
    );

    SetTargetFPS(60);
    SetRandomSeed((unsigned int)GetTime() * 1000000u);

    init_fluid_arrays();

    dt = 0.06f;
    rho = 1.0f;
    h = 1.0f;
    speed = 2;

    while (!WindowShouldClose()) {

        // Inject dye and momentum when the mouse is held.
        inject_mouse();

        // Advance the simulation every frame, even while drawing.
        for (int i = 0; i < speed; i++) {

            advect_velocity();
            apply_wall_boundaries();

            project_velocity();

            advect_dye();
        }

        BeginDrawing();

        ClearBackground(BLACK);
        render_fluid();

        EndDrawing();
    }

    CloseWindow();

    return 0;
}
