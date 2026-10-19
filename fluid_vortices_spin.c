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
#define PRESSURE_ITERATIONS 80
#define INFLOW_SPEED 0.65f
#define VORTEX_RADIUS 72.0f
#define VORTEX_STRENGTH 72.0f
#define INLET_DYE_RATE 0.22f

// Dye fading: higher means faster disappearance.
#define DYE_FADE_RATE 0.01f

static float dt;
static float rho;
static float h;
static int speed;

// Each field has two backing buffers. The field pointers are swapped after
// each update, avoiding full-grid memcpy operations.
static float u_data[N][N], u_tmp_data[N][N];
static float v_data[N][N], v_tmp_data[N][N];
static float (*u)[N] = u_data, (*u_tmp)[N] = u_tmp_data;
static float (*v)[N] = v_data, (*v_tmp)[N] = v_tmp_data;

static float pressure_data[N][N], pressure_tmp_data[N][N];
static float (*pressure)[N] = pressure_data;
static float (*pressure_tmp)[N] = pressure_tmp_data;
static float divergence[N][N];

// Dye concentration and average colour, also double-buffered via pointers.
static float dye_data[N][N], dye_tmp_data[N][N];
static float dye_r_data[N][N], dye_r_tmp_data[N][N];
static float dye_g_data[N][N], dye_g_tmp_data[N][N];
static float dye_b_data[N][N], dye_b_tmp_data[N][N];
static float (*dye)[N] = dye_data, (*dye_tmp)[N] = dye_tmp_data;
static float (*dye_r)[N] = dye_r_data, (*dye_r_tmp)[N] = dye_r_tmp_data;
static float (*dye_g)[N] = dye_g_data, (*dye_g_tmp)[N] = dye_g_tmp_data;
static float (*dye_b)[N] = dye_b_data, (*dye_b_tmp)[N] = dye_b_tmp_data;

// Render the entire simulation through one texture instead of thousands of draw calls.
static Color render_pixels[N * N];
static Texture2D fluid_texture;
static unsigned char solid[N][N];

typedef struct { float x, y, radius; float direction; } VortexGenerator;
// Counter-rotating generators: upper clockwise, lower counter-clockwise.
static const VortexGenerator vortices[2] = {
    {150.0f, 145.0f, VORTEX_RADIUS, -1.0f},
    {150.0f, 255.0f, VORTEX_RADIUS,  1.0f}
};
static bool velocity_colours = true;


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

        // Pick a saturated but deliberately subdued colour for each new stroke.
        // Limiting channel values keeps the dye from washing out the arrows.
        static const Color palette[] = {
            {190, 65, 75, 255}, {65, 145, 195, 255},
            {175, 115, 45, 255}, {130, 80, 175, 255},
            {45, 155, 115, 255}, {190, 90, 145, 255},
            {145, 155, 55, 255}, {65, 125, 175, 255}
        };
        drag_color = palette[GetRandomValue(0, (int)(sizeof(palette)/sizeof(palette[0])) - 1)];
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
// Wind-tunnel obstacles and boundary conditions
// --------------------------------------------------

static void rebuild_solid_mask(void)
{
    // The vortex generators are force fields, not solid obstacles.
    memset(solid, 0, sizeof(solid));
}

static void enforce_wind_tunnel(void)
{
    for (int y = 0; y < N; ++y) {
        // Left inlet and right outlet maintain a steady left-to-right stream.
        if (!solid[y][1]) u[y][1] = INFLOW_SPEED;
        if (!solid[y][0]) u[y][0] = INFLOW_SPEED;
        u[y][N - 1] = u[y][N - 2];
        v[y][0] = 0.0f;
        v[y][N - 1] = v[y][N - 2];
    }

    // Solid top and bottom walls.
    for (int x = 0; x < N; ++x) {
        v[0][x] = 0.0f;
        v[N - 1][x] = 0.0f;
        u[0][x] = 0.0f;
        u[N - 1][x] = 0.0f;
    }
}

static void handle_obstacle_input(void)
{
    // Vortex locations are fixed; V toggles velocity heatmap / dye view.
    if (IsKeyPressed(KEY_V)) velocity_colours = !velocity_colours;
}

// Apply localized tangential acceleration to create a pair of opposite vortices.
// Work only inside each generator's bounding box to keep the extra cost small.
static void apply_vortex_generators(void)
{
    const float force = VORTEX_STRENGTH * dt;
    for (int k = 0; k < 2; ++k) {
        const VortexGenerator *g = &vortices[k];
        int xmin = (int)fmaxf(1.0f, g->x - g->radius);
        int xmax = (int)fminf(N - 2.0f, g->x + g->radius);
        int ymin = (int)fmaxf(1.0f, g->y - g->radius);
        int ymax = (int)fminf(N - 2.0f, g->y + g->radius);
        const float inv_r = 1.0f / g->radius;

        for (int y = ymin; y <= ymax; ++y) {
            for (int x = xmin; x <= xmax; ++x) {
                float dx = (float)x - g->x;
                float dy = (float)y - g->y;
                float r2 = dx*dx + dy*dy;
                if (r2 >= g->radius*g->radius) continue;

                // Smooth falloff avoids a hard edge at the generator boundary.
                float q = 1.0f - sqrtf(r2) * inv_r;
                float strength = force * q * q * g->direction;
                u[y][x] += -dy * inv_r * strength;
                v[y][x] +=  dx * inv_r * strength;
            }
        }
    }
}

// Continuously inject a thin cyan tracer stream at the left inlet.
static void inject_inlet_dye(void)
{
    const float amount = INLET_DYE_RATE;
    for (int y = 3; y < N - 3; ++y) {
        // A narrow, gently pulsing band makes the incoming stream visible.
        if ((y % 48) < 3) continue;
        const int x = 2;
        float old = dye[y][x];
        float total = old + amount;
        if (total > 1.0f) total = 1.0f;
        float denom = old + amount;
        if (denom > 0.000001f) {
            dye_r[y][x] = (dye_r[y][x] * old + 0.10f * amount) / denom;
            dye_g[y][x] = (dye_g[y][x] * old + 0.85f * amount) / denom;
            dye_b[y][x] = (dye_b[y][x] * old + 1.00f * amount) / denom;
        }
        dye[y][x] = total;
    }
}

static Color velocity_colour(float ux, float vy)
{
    // Single-hue speed map: navy -> blue -> cyan -> near-white.
    // Direction is shown by arrows, not encoded as a rainbow.
    float mag = sqrtf(ux * ux + vy * vy);
    float t = fminf(1.0f, mag / (INFLOW_SPEED * 1.8f));

    if (t < 0.33f) {
        float q = t / 0.33f;
        return (Color){
            (unsigned char)(5 + 5*q),
            (unsigned char)(10 + 35*q),
            (unsigned char)(24 + 100*q), 255
        };
    } else if (t < 0.72f) {
        float q = (t - 0.33f) / 0.39f;
        return (Color){
            (unsigned char)(10 + 5*q),
            (unsigned char)(45 + 150*q),
            (unsigned char)(124 + 115*q), 255
        };
    } else {
        float q = (t - 0.72f) / 0.28f;
        return (Color){
            (unsigned char)(15 + 180*q),
            (unsigned char)(195 + 60*q),
            (unsigned char)(239 + 16*q), 255
        };
    }
}

// --------------------------------------------------
// Advect velocity using backward tracing
// --------------------------------------------------

static void advect_velocity(void)
{
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            if (x == 0 || x == N - 1 || y == 0 || y == N - 1 || solid[y][x]) {
                u_tmp[y][x] = u[y][x];
                v_tmp[y][x] = v[y][x];
                continue;
            }
            float xb = x - u[y][x] * dt / h;
            float yb = y - v[y][x] * dt / h;
            u_tmp[y][x] = sample(u, xb, yb);
            v_tmp[y][x] = sample(v, xb, yb);
        }
    }
    float (*swap_u)[N] = u; u = u_tmp; u_tmp = swap_u;
    float (*swap_v)[N] = v; v = v_tmp; v_tmp = swap_v;
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
    memset(pressure_data, 0, sizeof(pressure_data));
    memset(pressure_tmp_data, 0, sizeof(pressure_tmp_data));
    pressure = pressure_data;
    pressure_tmp = pressure_tmp_data;

    const float scale = rho * h * h / dt;
    float (*p)[N] = pressure;
    float (*p_next)[N] = pressure_tmp;

    for (int iter = 0; iter < PRESSURE_ITERATIONS; iter++) {
        for (int y = 1; y < N - 1; y++) {
            for (int x = 1; x < N - 1; x++) {
                if (solid[y][x]) { p_next[y][x] = 0.0f; continue; }
                float pc = p[y][x];
                float pl = solid[y][x-1] ? pc : p[y][x-1];
                float pr = solid[y][x+1] ? pc : p[y][x+1];
                float pu = solid[y-1][x] ? pc : p[y-1][x];
                float pd = solid[y+1][x] ? pc : p[y+1][x];
                p_next[y][x] = (pl + pr + pu + pd - scale * divergence[y][x]) * 0.25f;
            }
        }

        // Approximate zero-normal pressure gradient.
        for (int x = 0; x < N; x++) {
            p_next[0][x] = p_next[1][x];
            p_next[N - 1][x] = p_next[N - 2][x];
        }
        for (int y = 0; y < N; y++) {
            p_next[y][0] = p_next[y][1];
            p_next[y][N - 1] = p_next[y][N - 2];
        }

        // Swap buffers instead of copying all 160,000 floats every iteration.
        float (*swap)[N] = p;
        p = p_next;
        p_next = swap;
    }

    // Keep the global pressure pointer on the final Jacobi buffer; no copy.
    pressure = p;
    pressure_tmp = p_next;
}

// --------------------------------------------------
// Correct velocity using the pressure gradient
// --------------------------------------------------

static void correct_velocity(void)
{
    const float factor = dt / (rho * h);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            if (x == 0 || x == N - 1 || y == 0 || y == N - 1 || solid[y][x]) {
                u_tmp[y][x] = u[y][x];
                v_tmp[y][x] = v[y][x];
                continue;
            }
            float dpdx = pressure[y][x + 1] - pressure[y][x];
            float dpdy = pressure[y + 1][x] - pressure[y][x];
            u_tmp[y][x] = u[y][x] - factor * dpdx;
            v_tmp[y][x] = v[y][x] - factor * dpdy;
        }
    }
    float (*swap_u)[N] = u; u = u_tmp; u_tmp = swap_u;
    float (*swap_v)[N] = v; v = v_tmp; v_tmp = swap_v;
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
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            if (solid[y][x]) {
                dye_tmp[y][x] = dye_r_tmp[y][x] = dye_g_tmp[y][x] = dye_b_tmp[y][x] = 0.0f;
                continue;
            }
            float xb = x - u[y][x] * dt / h;
            float yb = y - v[y][x] * dt / h;
            dye_tmp[y][x] = sample(dye, xb, yb);
            dye_r_tmp[y][x] = sample(dye_r, xb, yb);
            dye_g_tmp[y][x] = sample(dye_g, xb, yb);
            dye_b_tmp[y][x] = sample(dye_b, xb, yb);
        }
    }
    float (*swap_d)[N] = dye; dye = dye_tmp; dye_tmp = swap_d;
    float (*swap_r)[N] = dye_r; dye_r = dye_r_tmp; dye_r_tmp = swap_r;
    float (*swap_g)[N] = dye_g; dye_g = dye_g_tmp; dye_g_tmp = swap_g;
    float (*swap_b)[N] = dye_b; dye_b = dye_b_tmp; dye_b_tmp = swap_b;

    const float fade = expf(-DYE_FADE_RATE * dt);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            dye[y][x] *= fade;
            if (dye[y][x] < 0.001f) {
                dye[y][x] = dye_r[y][x] = dye_g[y][x] = dye_b[y][x] = 0.0f;
            }
        }
    }
}

// --------------------------------------------------
// Initialize arrays
// --------------------------------------------------

static void init_fluid_arrays(void)
{
    memset(u_data, 0, sizeof(u_data)); memset(u_tmp_data, 0, sizeof(u_tmp_data));
    memset(v_data, 0, sizeof(v_data)); memset(v_tmp_data, 0, sizeof(v_tmp_data));
    u = u_data; u_tmp = u_tmp_data; v = v_data; v_tmp = v_tmp_data;
    memset(pressure_data, 0, sizeof(pressure_data));
    memset(pressure_tmp_data, 0, sizeof(pressure_tmp_data));
    pressure = pressure_data; pressure_tmp = pressure_tmp_data;
    memset(divergence, 0, sizeof(divergence));
    memset(dye_data, 0, sizeof(dye_data)); memset(dye_tmp_data, 0, sizeof(dye_tmp_data));
    memset(dye_r_data, 0, sizeof(dye_r_data)); memset(dye_r_tmp_data, 0, sizeof(dye_r_tmp_data));
    memset(dye_g_data, 0, sizeof(dye_g_data)); memset(dye_g_tmp_data, 0, sizeof(dye_g_tmp_data));
    memset(dye_b_data, 0, sizeof(dye_b_data)); memset(dye_b_tmp_data, 0, sizeof(dye_b_tmp_data));
    dye = dye_data; dye_tmp = dye_tmp_data;
    dye_r = dye_r_data; dye_r_tmp = dye_r_tmp_data;
    dye_g = dye_g_data; dye_g_tmp = dye_g_tmp_data;
    dye_b = dye_b_data; dye_b_tmp = dye_b_tmp_data;
}

// --------------------------------------------------
// Render coloured dye
// Concentration controls opacity, not RGB brightness.
// --------------------------------------------------

static void render_fluid(void)
{
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int i = y * N + x;
            if (solid[y][x]) {
                render_pixels[i] = (Color){48, 52, 60, 255};
                continue;
            }

            Color base = velocity_colours
                ? velocity_colour(u[y][x], v[y][x])
                : (Color){5, 8, 16, 255};

            // Blend subdued dye into the speed map (or dark dye view).
            // Keep the dye RGB and blend amount capped so white arrows remain legible.
            float d = fmaxf(0.0f, fminf(1.0f, dye[y][x]));
            if (d > 0.005f) {
                float mix = fminf(0.52f, d * 0.65f);
                float dr = fminf(0.62f, fmaxf(0.0f, dye_r[y][x]));
                float dg = fminf(0.62f, fmaxf(0.0f, dye_g[y][x]));
                float db = fminf(0.62f, fmaxf(0.0f, dye_b[y][x]));
                base.r = (unsigned char)(base.r * (1.0f - mix) + dr * 255.0f * mix);
                base.g = (unsigned char)(base.g * (1.0f - mix) + dg * 255.0f * mix);
                base.b = (unsigned char)(base.b * (1.0f - mix) + db * 255.0f * mix);
            }
            render_pixels[i] = base;
        }
    }
    UpdateTexture(fluid_texture, render_pixels);
    DrawTextureEx(fluid_texture, (Vector2){0.0f, 0.0f}, 0.0f, CELL, WHITE);
}

// Draw a sparse velocity vector field over the coloured speed map.
// Sampling every 16 cells keeps draw-call overhead small (about 625 arrows).
static void draw_velocity_arrows(void)
{
    const int spacing = 16;
    for (int y = spacing / 2; y < N - 1; y += spacing) {
        for (int x = spacing / 2; x < N - 1; x += spacing) {
            float ux = u[y][x];
            float vy = v[y][x];
            float mag = sqrtf(ux * ux + vy * vy);
            if (mag < 0.08f) continue;

            float nx = ux / mag;
            float ny = vy / mag;
            float px = (x + 0.5f) * CELL;
            float py = (y + 0.5f) * CELL;
            float len = 4.0f + 8.0f * fminf(1.0f, mag / (INFLOW_SPEED * 1.5f));
            float ex = px + nx * len;
            float ey = py + ny * len;
            float head = 3.2f;

            // Small arrow head, with coordinates in screen space.
            float bx = ex - nx * head;
            float by = ey - ny * head;
            float pxh = -ny * head * 0.65f;
            float pyh =  nx * head * 0.65f;

            // Dark under-stroke gives each white arrow contrast on every dye colour.
            DrawLineEx((Vector2){px, py}, (Vector2){ex, ey}, 2.8f, (Color){0, 0, 0, 245});
            DrawLineEx((Vector2){ex, ey}, (Vector2){bx + pxh, by + pyh}, 2.8f, (Color){0, 0, 0, 245});
            DrawLineEx((Vector2){ex, ey}, (Vector2){bx - pxh, by - pyh}, 2.8f, (Color){0, 0, 0, 245});
            Color arrow = (Color){255, 255, 255, 255};
            DrawLineEx((Vector2){px, py}, (Vector2){ex, ey}, 1.25f, arrow);
            DrawLineEx((Vector2){ex, ey}, (Vector2){bx + pxh, by + pyh}, 1.25f, arrow);
            DrawLineEx((Vector2){ex, ey}, (Vector2){bx - pxh, by - pyh}, 1.25f, arrow);
        }
    }
}

static void draw_overlay(void)
{
    DrawText("VELOCITY: colour = speed | arrows = direction", 10, 8, 15, RAYWHITE);
    DrawText("Left-click/drag: inject coloured dye | V: toggle speed map / dye view", 10, SCREEN_H - 24, 13, RAYWHITE);
    for (int k = 0; k < 2; ++k) {
        const VortexGenerator *g = &vortices[k];
        Color c = k == 0 ? (Color){255, 125, 70, 255} : (Color){90, 170, 255, 255};
        DrawCircleLines((int)(g->x*CELL), (int)(g->y*CELL), g->radius*CELL, c);
        DrawCircle((int)(g->x*CELL), (int)(g->y*CELL), 3.0f, c);
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
        "Counter-Rotating Vortices - Inlet Flow"
    );

    Image render_image = GenImageColor(N, N, BLANK);
    fluid_texture = LoadTextureFromImage(render_image);
    UnloadImage(render_image);
    SetTextureFilter(fluid_texture, TEXTURE_FILTER_POINT);

    SetTargetFPS(60);
    SetRandomSeed((unsigned int)GetTime() * 1000000u);

    init_fluid_arrays();
    rebuild_solid_mask();
    // Start at rest. Fluid enters only through the left inlet during simulation.

    dt = 0.06f;
    rho = 1.0f;
    h = 1.0f;
    speed = 2;

    while (!WindowShouldClose()) {

        handle_obstacle_input();
        inject_mouse();

        // Keep the wind tunnel flowing while advancing the solver.
        for (int i = 0; i < speed; i++) {
            enforce_wind_tunnel();
            apply_vortex_generators();
            inject_inlet_dye();
            advect_velocity();
            enforce_wind_tunnel();
            project_velocity();
            enforce_wind_tunnel();
            advect_dye();
        }

        BeginDrawing();

        ClearBackground(BLACK);
        render_fluid();
        draw_velocity_arrows();
        draw_overlay();

        EndDrawing();
    }

    UnloadTexture(fluid_texture);
    CloseWindow();

    return 0;
}