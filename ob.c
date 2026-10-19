#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/*
 * 2-D wind-tunnel / Navier-Stokes showcase.
 *
 * Grid: 2 screen pixels per cell; the fluid strip is 120 px tall.
 * Numerical method: semi-Lagrangian advection, implicit viscosity (Jacobi),
 * and a pressure-projection step. Cell-centred velocities are used here for
 * simplicity; this is an educational/visual solver rather than production CFD.
 */

#define SCREEN_W 1280
#define SCREEN_H 800
#define CELL 2
#define CHANNEL_Y 335
#define CHANNEL_H 120
#define NX (SCREEN_W / CELL)
#define NY (CHANNEL_H / CELL)
#define DT (1.0f / 60.0f)
#define PRESSURE_ITERS 42
#define VISCOSITY_ITERS 10
#define MOUSE_RADIUS_PX 16.0f
#define MAX_OBSTACLES 24
#define PI 3.14159265358979323846f

#if defined(__GNUC__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

typedef struct { float r, g, b; } Dye;
typedef enum { SHAPE_CIRCLE, SHAPE_SQUARE, SHAPE_STAR } Shape;
typedef struct {
    Vector2 pos;       /* screen coordinates */
    float size;        /* circle radius / square half-width / star outer radius */
    float angle;       /* radians */
    Shape shape;
} Obstacle;

static float u[NY][NX], v[NY][NX];
static float advU[NY][NX], advV[NY][NX];
static float diffU[NY][NX], diffV[NY][NX], diffWork[NY][NX];
static float pressure[NY][NX], pressureNext[NY][NX], divergence[NY][NX];
static Dye dye[NY][NX], dyeNext[NY][NX];
static unsigned char solid[NY][NX], oldSolid[NY][NX];
static Obstacle obstacles[MAX_OBSTACLES];
static int obstacleCount = 0, selectedObstacle = -1;
static bool maskDirty = true;
static bool showHeat = true, showDye = true, paused = false;
static float viscosity = 3.0f;       /* grid-cell^2 / second */
static float inletSpeed = 42.0f;     /* grid cells / second */
static float dyeHue = 15.0f;
static Color activeDye = {255, 95, 45, 255};
static Vector2 previousMouse = {0};
static bool previousMouseValid = false;
static bool draggingObstacle = false;
static Vector2 dragOffset = {0};

static float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static float sqrf(float x) { return x * x; }

static int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static bool in_channel(float x, float y) {
    return x >= 0 && x < SCREEN_W && y >= CHANNEL_Y && y < CHANNEL_Y + CHANNEL_H;
}

static Vector2 channel_to_grid(float x, float y) {
    Vector2 p = { x / (float)CELL - 0.5f,
                  (y - CHANNEL_Y) / (float)CELL - 0.5f };
    return p;
}

static void star_vertex(const Obstacle *o, int i, Vector2 *out) {
    float radius = (i % 2 == 0) ? o->size : o->size * 0.43f;
    float a = -PI * 0.5f + o->angle + (float)i * PI / 5.0f;
    out->x = o->pos.x + cosf(a) * radius;
    out->y = o->pos.y + sinf(a) * radius;
}

/* Point test is performed in the obstacle's local coordinate system. */
static bool point_in_obstacle(const Obstacle *o, float px, float py) {
    float dx = px - o->pos.x;
    float dy = py - o->pos.y;
    float c = cosf(o->angle), s = sinf(o->angle);
    float lx =  c * dx + s * dy;
    float ly = -s * dx + c * dy;

    if (o->shape == SHAPE_CIRCLE)
        return lx * lx + ly * ly <= o->size * o->size;

    if (o->shape == SHAPE_SQUARE)
        return fabsf(lx) <= o->size && fabsf(ly) <= o->size;

    /* Five-point concave star polygon, ray-cast in the unrotated local frame. */
    bool inside = false;
    Vector2 verts[10];
    for (int i = 0; i < 10; ++i) {
        float r = (i % 2 == 0) ? o->size : o->size * 0.43f;
        float a = -PI * 0.5f + (float)i * PI / 5.0f;
        verts[i] = (Vector2){ cosf(a) * r, sinf(a) * r };
    }
    for (int i = 0, j = 9; i < 10; j = i++) {
        bool crosses = ((verts[i].y > ly) != (verts[j].y > ly)) &&
            (lx < (verts[j].x - verts[i].x) * (ly - verts[i].y) /
                   ((verts[j].y - verts[i].y) + 1.0e-20f) + verts[i].x);
        if (crosses) inside = !inside;
    }
    return inside;
}

static void set_active_dye(void) {
    activeDye = ColorFromHSV(dyeHue, 0.78f, 1.0f);
    dyeHue += 67.0f;
    if (dyeHue >= 360.0f) dyeHue -= 360.0f;
}

static void add_obstacle(Shape shape, Vector2 pos) {
    if (obstacleCount >= MAX_OBSTACLES) return;
    pos.x = clampf(pos.x, 24.0f, SCREEN_W - 24.0f);
    pos.y = clampf(pos.y, CHANNEL_Y + 20.0f, CHANNEL_Y + CHANNEL_H - 20.0f);
    Obstacle o = {0};
    o.pos = pos;
    o.shape = shape;
    o.size = shape == SHAPE_CIRCLE ? 17.0f : (shape == SHAPE_SQUARE ? 16.0f : 19.0f);
    o.angle = 0.0f;
    obstacles[obstacleCount] = o;
    selectedObstacle = obstacleCount++;
    maskDirty = true;
}

static void rebuild_solid_mask(void) {
    memcpy(oldSolid, solid, sizeof(solid));
    memset(solid, 0, sizeof(solid));

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            float px = (x + 0.5f) * CELL;
            float py = CHANNEL_Y + (y + 0.5f) * CELL;
            for (int k = 0; k < obstacleCount; ++k) {
                if (point_in_obstacle(&obstacles[k], px, py)) {
                    solid[y][x] = 1;
                    break;
                }
            }
        }
    }

    /* Newly uncovered cells are initialized with the background stream. */
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (solid[y][x]) {
                u[y][x] = v[y][x] = 0.0f;
                dye[y][x] = (Dye){0, 0, 0};
            } else if (oldSolid[y][x]) {
                u[y][x] = inletSpeed;
                v[y][x] = 0.0f;
                dye[y][x] = (Dye){0, 0, 0};
            }
        }
    }
    maskDirty = false;
}

/* Bilinear scalar sampler that ignores solid cells and renormalizes weights. */
static float sample_scalar(const float field[NY][NX], float x, float y) {
    x = clampf(x, 0.0f, (float)(NX - 1));
    y = clampf(y, 0.0f, (float)(NY - 1));
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    int x1 = clampi(x0 + 1, 0, NX - 1);
    int y1 = clampi(y0 + 1, 0, NY - 1);
    float tx = x - x0, ty = y - y0;
    float weights[4] = {(1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty};
    int xs[4] = {x0, x1, x0, x1};
    int ys[4] = {y0, y0, y1, y1};
    float sum = 0.0f, total = 0.0f;
    for (int i = 0; i < 4; ++i) {
        if (!solid[ys[i]][xs[i]]) {
            sum += field[ys[i]][xs[i]] * weights[i];
            total += weights[i];
        }
    }
    if (total < 1.0e-6f) return 0.0f;
    return sum / total;
}

static Dye sample_dye(float x, float y) {
    x = clampf(x, 0.0f, (float)(NX - 1));
    y = clampf(y, 0.0f, (float)(NY - 1));
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    int x1 = clampi(x0 + 1, 0, NX - 1);
    int y1 = clampi(y0 + 1, 0, NY - 1);
    float tx = x - x0, ty = y - y0;
    float weights[4] = {(1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty};
    int xs[4] = {x0, x1, x0, x1};
    int ys[4] = {y0, y0, y1, y1};
    Dye result = {0, 0, 0};
    float total = 0.0f;
    for (int i = 0; i < 4; ++i) {
        if (solid[ys[i]][xs[i]]) continue;
        Dye d = dye[ys[i]][xs[i]];
        result.r += d.r * weights[i];
        result.g += d.g * weights[i];
        result.b += d.b * weights[i];
        total += weights[i];
    }
    if (total > 1.0e-6f) {
        result.r /= total; result.g /= total; result.b /= total;
    }
    return result;
}

static void apply_inlet(void) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < 3; ++x) {
            if (!solid[y][x]) {
                u[y][x] = inletSpeed;
                v[y][x] *= 0.15f;
            }
        }
    }
    /* Open, zero-gradient outlet. */
    for (int y = 0; y < NY; ++y) {
        if (!solid[y][NX-1] && !solid[y][NX-2]) {
            u[y][NX-1] = u[y][NX-2];
            v[y][NX-1] = v[y][NX-2];
        }
    }
}

/* No-penetration approximation for cell-centred velocities near obstacles. */
static void enforce_walls(void) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (solid[y][x]) {
                u[y][x] = v[y][x] = 0.0f;
                continue;
            }
            if (x == 0 || solid[y][x-1] || (x+1 < NX && solid[y][x+1])) u[y][x] = 0.0f;
            if (y == 0 || (y > 0 && solid[y-1][x]) || y == NY-1 || (y+1 < NY && solid[y+1][x])) v[y][x] = 0.0f;
        }
    }
}

static void advect_velocity(void) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (solid[y][x]) {
                advU[y][x] = advV[y][x] = 0.0f;
                continue;
            }
            float backX = x - u[y][x] * DT;
            float backY = y - v[y][x] * DT;
            advU[y][x] = sample_scalar(u, backX, backY);
            advV[y][x] = sample_scalar(v, backX, backY);
        }
    }
    memcpy(u, advU, sizeof(u));
    memcpy(v, advV, sizeof(v));
}

/* Correct Jacobi diffusion: each iteration reads dst and writes work. */
static void diffuse_field(const float src[NY][NX], float dst[NY][NX], float alpha) {
    memcpy(dst, src, sizeof(u));
    if (alpha <= 1.0e-7f) return;
    for (int iter = 0; iter < VISCOSITY_ITERS; ++iter) {
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                if (solid[y][x]) { diffWork[y][x] = 0.0f; continue; }
                float sum = 0.0f;
                int count = 0;
                const int dx[4] = {-1, 1, 0, 0};
                const int dy[4] = {0, 0, -1, 1};
                for (int n = 0; n < 4; ++n) {
                    int xx = x + dx[n], yy = y + dy[n];
                    if (xx < 0 || xx >= NX || yy < 0 || yy >= NY || solid[yy][xx]) continue;
                    sum += dst[yy][xx];
                    count++;
                }
                diffWork[y][x] = (src[y][x] + alpha * sum) / (1.0f + alpha * count);
            }
        }
        memcpy(dst, diffWork, sizeof(u));
    }
}

static float neighbor_pressure(int x, int y, float center) {
    if (x < 0 || x >= NX || y < 0 || y >= NY || solid[y][x]) return center;
    return pressure[y][x];
}

static void project_velocity(void) {
    /* Divergence in grid units / second. Solid neighbours act as wall faces. */
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (solid[y][x]) { divergence[y][x] = pressure[y][x] = 0.0f; continue; }
            float ue = (x+1 < NX && !solid[y][x+1]) ? u[y][x+1] : u[y][x];
            float uw = (x > 0 && !solid[y][x-1]) ? u[y][x-1] : u[y][x];
            float vs = (y+1 < NY && !solid[y+1][x]) ? v[y+1][x] : v[y][x];
            float vn = (y > 0 && !solid[y-1][x]) ? v[y-1][x] : v[y][x];
            divergence[y][x] = 0.5f * ((ue - uw) + (vs - vn));
            pressure[y][x] = 0.0f;
        }
    }

    for (int iter = 0; iter < PRESSURE_ITERS; ++iter) {
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                if (solid[y][x]) { pressureNext[y][x] = 0.0f; continue; }
                float sum = 0.0f;
                int count = 0;
                const int dx[4] = {-1, 1, 0, 0};
                const int dy[4] = {0, 0, -1, 1};
                for (int n = 0; n < 4; ++n) {
                    int xx = x + dx[n], yy = y + dy[n];
                    if (xx < 0 || xx >= NX || yy < 0 || yy >= NY || solid[yy][xx]) continue;
                    sum += pressure[yy][xx];
                    count++;
                }
                pressureNext[y][x] = count ? (sum - divergence[y][x] / DT) / (float)count : 0.0f;
            }
        }
        memcpy(pressure, pressureNext, sizeof(pressure));
    }

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (solid[y][x]) { u[y][x] = v[y][x] = 0.0f; continue; }
            float pc = pressure[y][x];
            float pe = neighbor_pressure(x+1, y, pc);
            float pw = neighbor_pressure(x-1, y, pc);
            float ps = neighbor_pressure(x, y+1, pc);
            float pn = neighbor_pressure(x, y-1, pc);
            u[y][x] -= 0.5f * DT * (pe - pw);
            v[y][x] -= 0.5f * DT * (ps - pn);
        }
    }
    enforce_walls();
}

static void advect_dye(void) {
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if (solid[y][x]) { dyeNext[y][x] = (Dye){0, 0, 0}; continue; }
            float backX = x - u[y][x] * DT;
            float backY = y - v[y][x] * DT;
            Dye d = sample_dye(backX, backY);
            d.r *= 0.9985f; d.g *= 0.9985f; d.b *= 0.9985f;
            dyeNext[y][x] = d;
        }
    }
    memcpy(dye, dyeNext, sizeof(dye));
}

static void inject_at(float px, float py, float mouseDx, float mouseDy) {
    if (!in_channel(px, py)) return;
    Vector2 g = channel_to_grid(px, py);
    int cx = (int)lroundf(g.x), cy = (int)lroundf(g.y);
    int radius = (int)ceilf(MOUSE_RADIUS_PX / CELL);
    float kickU = clampf((mouseDx / CELL) / DT * 0.30f, -150.0f, 150.0f);
    float kickV = clampf((mouseDy / CELL) / DT * 0.30f, -150.0f, 150.0f);
    float cr = activeDye.r / 255.0f, cg = activeDye.g / 255.0f, cb = activeDye.b / 255.0f;

    for (int y = cy - radius; y <= cy + radius; ++y) {
        if (y < 0 || y >= NY) continue;
        for (int x = cx - radius; x <= cx + radius; ++x) {
            if (x < 0 || x >= NX || solid[y][x]) continue;
            float dx = x - g.x, dy = y - g.y;
            float distPx = sqrtf(dx*dx + dy*dy) * CELL;
            if (distPx > MOUSE_RADIUS_PX) continue;
            float weight = 1.0f - distPx / MOUSE_RADIUS_PX;
            float weight2 = weight * weight;
            u[y][x] += kickU * weight2;
            v[y][x] += kickV * weight2;
            dye[y][x].r = clampf(dye[y][x].r + cr * 0.9f * weight, 0.0f, 1.0f);
            dye[y][x].g = clampf(dye[y][x].g + cg * 0.9f * weight, 0.0f, 1.0f);
            dye[y][x].b = clampf(dye[y][x].b + cb * 0.9f * weight, 0.0f, 1.0f);
        }
    }
}

static Color velocity_color(float speed) {
    /* Blue -> cyan -> green -> yellow -> red as speed increases. */
    float t = clampf(speed / 90.0f, 0.0f, 1.0f);
    return ColorFromHSV((1.0f - t) * 240.0f, 0.94f, 0.95f);
}

static Color mix_color(Color a, Color b, float t) {
    t = clampf(t, 0.0f, 1.0f);
    return (Color){
        (unsigned char)(a.r * (1.0f-t) + b.r*t),
        (unsigned char)(a.g * (1.0f-t) + b.g*t),
        (unsigned char)(a.b * (1.0f-t) + b.b*t), 255
    };
}

static void draw_obstacle(const Obstacle *o, bool selected) {
    /* Keep every obstacle the same fill colour. Selection is indicated by its
       outline only, so circles, squares and stars have a consistent appearance. */
    Color fill = (Color){226, 232, 240, 255};
    Color outline = selected ? ORANGE : (Color){20, 27, 38, 255};

    if (o->shape == SHAPE_CIRCLE) {
        DrawCircleV(o->pos, o->size, fill);
        DrawCircleLines((int)o->pos.x, (int)o->pos.y, o->size, outline);
        return;
    }
    if (o->shape == SHAPE_SQUARE) {
        /* Build one set of world-space corners for fill, outline, and mask logic.
           Avoid DrawRectanglePro here: its pivot convention made the visual fill
           appear offset from the rotated solid-cell mask on some Raylib versions. */
        const Vector2 local[4] = {
            {-o->size, -o->size}, {o->size, -o->size},
            {o->size, o->size}, {-o->size, o->size}
        };
        Vector2 p[4];
        float c = cosf(o->angle), s = sinf(o->angle);
        for (int i = 0; i < 4; ++i) {
            p[i] = (Vector2){
                o->pos.x + c * local[i].x - s * local[i].y,
                o->pos.y + s * local[i].x + c * local[i].y
            };
        }
        DrawTriangle(p[0], p[1], p[2], fill);
        DrawTriangle(p[0], p[2], p[3], fill);
        for (int i = 0; i < 4; ++i)
            DrawLineEx(p[i], p[(i + 1) % 4], selected ? 2.5f : 1.5f, outline);
        return;
    }

    Vector2 pts[10];
    for (int i = 0; i < 10; ++i) star_vertex(o, i, &pts[i]);
    for (int i = 0; i < 10; ++i) {
        Vector2 a = pts[i], b = pts[(i+1)%10];
        DrawTriangle(o->pos, a, b, fill);
        DrawLineEx(a, b, selected ? 2.5f : 1.5f, outline);
    }
}

static int hit_obstacle(float px, float py) {
    for (int i = obstacleCount - 1; i >= 0; --i) {
        if (point_in_obstacle(&obstacles[i], px, py)) return i;
    }
    return -1;
}

static void reset_simulation(void) {
    memset(u, 0, sizeof(u)); memset(v, 0, sizeof(v));
    memset(advU, 0, sizeof(advU)); memset(advV, 0, sizeof(advV));
    memset(pressure, 0, sizeof(pressure)); memset(pressureNext, 0, sizeof(pressureNext));
    memset(divergence, 0, sizeof(divergence)); memset(dye, 0, sizeof(dye));
    for (int y = 0; y < NY; ++y)
        for (int x = 0; x < NX; ++x)
            if (!solid[y][x]) u[y][x] = inletSpeed;
    apply_inlet();
}

int main(void) {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT);
    InitWindow(SCREEN_W, SCREEN_H, "Raylib 2D Navier-Stokes Wind Tunnel");
    SetTargetFPS(60);

    add_obstacle(SHAPE_CIRCLE, (Vector2){355, CHANNEL_Y + CHANNEL_H*0.50f});
    add_obstacle(SHAPE_SQUARE, (Vector2){650, CHANNEL_Y + CHANNEL_H*0.50f});
    obstacles[1].angle = 0.20f;
    add_obstacle(SHAPE_STAR, (Vector2){930, CHANNEL_Y + CHANNEL_H*0.50f});
    selectedObstacle = -1;
    rebuild_solid_mask();
    reset_simulation();
    set_active_dye();

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();
        bool mouseInChannel = in_channel(mouse.x, mouse.y);

        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_V)) showHeat = !showHeat;
        if (IsKeyPressed(KEY_D)) showDye = !showDye;
        if (IsKeyPressed(KEY_R)) reset_simulation();
        if (IsKeyPressed(KEY_UP)) viscosity = clampf(viscosity + 0.5f, 0.0f, 10.0f);
        if (IsKeyPressed(KEY_DOWN)) viscosity = clampf(viscosity - 0.5f, 0.0f, 10.0f);
        if (IsKeyPressed(KEY_RIGHT)) inletSpeed = clampf(inletSpeed + 3.0f, 9.0f, 72.0f);
        if (IsKeyPressed(KEY_LEFT)) inletSpeed = clampf(inletSpeed - 3.0f, 9.0f, 72.0f);

        if (IsKeyPressed(KEY_C)) add_obstacle(SHAPE_CIRCLE, mouseInChannel ? mouse : (Vector2){SCREEN_W*0.50f, CHANNEL_Y+CHANNEL_H*0.5f});
        if (IsKeyPressed(KEY_S)) add_obstacle(SHAPE_SQUARE, mouseInChannel ? mouse : (Vector2){SCREEN_W*0.50f, CHANNEL_Y+CHANNEL_H*0.5f});
        if (IsKeyPressed(KEY_T)) add_obstacle(SHAPE_STAR, mouseInChannel ? mouse : (Vector2){SCREEN_W*0.50f, CHANNEL_Y+CHANNEL_H*0.5f});

        if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
            selectedObstacle = hit_obstacle(mouse.x, mouse.y);
            draggingObstacle = selectedObstacle >= 0;
            if (draggingObstacle) {
                dragOffset = (Vector2){obstacles[selectedObstacle].pos.x - mouse.x,
                                       obstacles[selectedObstacle].pos.y - mouse.y};
            }
        }
        if (draggingObstacle && IsMouseButtonDown(MOUSE_BUTTON_RIGHT) && selectedObstacle >= 0) {
            obstacles[selectedObstacle].pos.x = clampf(mouse.x + dragOffset.x, 20.0f, SCREEN_W - 20.0f);
            obstacles[selectedObstacle].pos.y = clampf(mouse.y + dragOffset.y, CHANNEL_Y + 18.0f, CHANNEL_Y + CHANNEL_H - 18.0f);
            maskDirty = true;
        }
        if (IsMouseButtonReleased(MOUSE_BUTTON_RIGHT)) draggingObstacle = false;
        float wheel = GetMouseWheelMove();
        if (selectedObstacle >= 0 && fabsf(wheel) > 0.0f) {
            obstacles[selectedObstacle].angle += wheel * 0.10f;
            maskDirty = true;
        }
        if (IsKeyPressed(KEY_X) && selectedObstacle >= 0) {
            for (int i = selectedObstacle; i < obstacleCount - 1; ++i) obstacles[i] = obstacles[i+1];
            obstacleCount--;
            selectedObstacle = -1;
            maskDirty = true;
        }

        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) set_active_dye();
        if (maskDirty) rebuild_solid_mask();

        if (!paused) {
            apply_inlet();
            if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && mouseInChannel) {
                if (!previousMouseValid) previousMouse = mouse;
                float dx = mouse.x - previousMouse.x;
                float dy = mouse.y - previousMouse.y;
                float dist = sqrtf(dx*dx + dy*dy);
                int steps = clampi((int)ceilf(dist / (CELL * 0.65f)), 1, 48);
                for (int i = 0; i <= steps; ++i) {
                    float t = (float)i / steps;
                    inject_at(previousMouse.x + dx*t, previousMouse.y + dy*t, dx, dy);
                }
                previousMouse = mouse;
                previousMouseValid = true;
            } else {
                previousMouseValid = false;
            }

            advect_velocity();
            float alpha = viscosity * DT;
            diffuse_field(u, diffU, alpha); memcpy(u, diffU, sizeof(u));
            diffuse_field(v, diffV, alpha); memcpy(v, diffV, sizeof(v));
            enforce_walls();
            project_velocity();
            apply_inlet();
            advect_dye();
        }

        BeginDrawing();
        ClearBackground((Color){12, 17, 27, 255});

        DrawText("2D NAVIER-STOKES  /  WIND TUNNEL", 28, 20, 26, RAYWHITE);
        DrawText("Left-drag: inject coloured dye + momentum    Right-drag: move obstacle    Scroll: rotate selected obstacle",
                 30, 58, 16, (Color){177, 190, 210, 255});
        DrawText(TextFormat("Grid %d x %d  |  cell %d px  |  viscosity %.1f  |  inlet %.0f cells/s",
                            NX, NY, CELL, viscosity, inletSpeed),
                 30, 86, 17, (Color){137, 202, 232, 255});

        /* Streamline channel backdrop and pixel-level velocity heatmap. */
        DrawRectangle(0, CHANNEL_Y - 7, SCREEN_W, CHANNEL_H + 14, (Color){31, 43, 60, 255});
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                Color col;
                if (solid[y][x]) {
                    /* Render the solid-cell mask using the same light fill as
                       the drawn obstacle. This guarantees square/star interiors
                       remain visible even on systems where triangle fills differ. */
                    col = (Color){226, 232, 240, 255};
                } else {
                    float speed = sqrtf(u[y][x]*u[y][x] + v[y][x]*v[y][x]);
                    col = showHeat ? velocity_color(speed) : (Color){28, 55, 77, 255};
                    if (showDye) {
                        Dye d = dye[y][x];
                        float amt = clampf(fmaxf(d.r, fmaxf(d.g, d.b)), 0.0f, 1.0f);
                        Color ink = {(unsigned char)(clampf(d.r,0,1)*255),
                                     (unsigned char)(clampf(d.g,0,1)*255),
                                     (unsigned char)(clampf(d.b,0,1)*255), 255};
                        col = mix_color(col, ink, amt * 0.62f);
                    }
                }
                DrawRectangle(x*CELL, CHANNEL_Y + y*CELL, CELL, CELL, col);
            }
        }

        /* Top and bottom channel walls. */
        DrawRectangle(0, CHANNEL_Y - 7, SCREEN_W, 5, (Color){92, 113, 139, 255});
        DrawRectangle(0, CHANNEL_Y + CHANNEL_H + 2, SCREEN_W, 5, (Color){92, 113, 139, 255});
        DrawText("INLET  ->", 13, CHANNEL_Y - 29, 15, (Color){135, 213, 243, 255});
        DrawText("OUTLET  ->", SCREEN_W - 112, CHANNEL_Y - 29, 15, (Color){135, 213, 243, 255});

        for (int i = 0; i < obstacleCount; ++i) draw_obstacle(&obstacles[i], i == selectedObstacle);

        /* Compact heatmap legend. */
        DrawText("VELOCITY", 30, CHANNEL_Y + CHANNEL_H + 25, 15, RAYWHITE);
        const int legendX = 128, legendY = CHANNEL_Y + CHANNEL_H + 28;
        for (int i = 0; i < 180; ++i) {
            Color c = ColorFromHSV((1.0f - i/179.0f) * 240.0f, 0.94f, 0.95f);
            DrawRectangle(legendX + i, legendY, 1, 12, c);
        }
        DrawText("LOW", legendX - 1, legendY + 16, 12, (Color){170, 184, 202, 255});
        DrawText("HIGH", legendX + 151, legendY + 16, 12, (Color){170, 184, 202, 255});
        DrawText("C circle   S square   T star   X delete   V heatmap   D dye   Space pause   R reset",
                 30, CHANNEL_Y + CHANNEL_H + 73, 16, (Color){203, 213, 225, 255});
        DrawText("UP/DOWN viscosity     LEFT/RIGHT inlet speed", 30, CHANNEL_Y + CHANNEL_H + 99, 16,
                 (Color){139, 158, 182, 255});
        DrawText(paused ? "PAUSED" : "RUNNING", SCREEN_W - 120, 20, 16,
                 paused ? ORANGE : (Color){114, 235, 164, 255});

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
