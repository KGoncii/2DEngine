import pygame
import random
import math

WIDTH, HEIGHT = 800, 600
FPS = 60

FRICTION_COEFF = 0.1
AIR_RESISTANCE = 1
ANGULAR_RESISTANCE = 1
MAX_ANGULAR_VELOCITY = 3
REPULSE_RADIUS = 200
REPULSE_FORCE = 10000
DENSITY = 0.01

show_triangulation = False

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Generowanie losowego prostego wielokąta (przybliżony konweks)
def generate_random_polygon(center_x, center_y, radius, vertex_count):
    angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(vertex_count)])
    points = []
    for angle in angles:
        r = random.uniform(radius * 0.7, radius)
        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        points.append(pygame.Vector2(x, y))
    return points

# Triangulacja Ear Clipping
def ear_clip_triangulation(poly):
    triangles = []
    vertices = poly[:]
    while len(vertices) > 3:
        ear_found = False
        n = len(vertices)
        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n

            a, b, c = vertices[prev_idx], vertices[i], vertices[next_idx]

            ab = b - a
            ac = c - a
            if ab.cross(ac) <= 0:
                continue

            triangle = [a, b, c]

            def point_in_triangle(p, tri):
                a, b, c = tri
                v0 = c - a
                v1 = b - a
                v2 = p - a
                dot00 = v0.dot(v0)
                dot01 = v0.dot(v1)
                dot02 = v0.dot(v2)
                dot11 = v1.dot(v1)
                dot12 = v1.dot(v2)
                denom = dot00 * dot11 - dot01 * dot01
                if denom == 0:
                    return False
                inv_denom = 1 / denom
                u = (dot11 * dot02 - dot01 * dot12) * inv_denom
                v = (dot00 * dot12 - dot01 * dot02) * inv_denom
                return (u >= 0) and (v >= 0) and (u + v < 1)

            if any(point_in_triangle(p, triangle) for j, p in enumerate(vertices) if j not in (prev_idx, i, next_idx)):
                continue

            triangles.append((a, b, c))
            del vertices[i]
            ear_found = True
            break
        if not ear_found:
            break
    if len(vertices) == 3:
        triangles.append((vertices[0], vertices[1], vertices[2]))
    return triangles

def polygon_area(points):
    area = 0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i].x * points[j].y - points[j].x * points[i].y
    return abs(area) / 2

class PolygonBody:
    DENSITY = DENSITY
    def __init__(self, points, mass, vx=0, vy=0, angle=0, angular_velocity=0):
        self.points = points
        self.mass = PolygonBody.DENSITY * polygon_area(points)
        self.pos = sum(points, pygame.Vector2(0, 0)) / len(points)
        self.vel = pygame.Vector2(vx, vy)
        self.angle = angle
        self.angular_velocity = angular_velocity
        self.inertia = self.mass * max((p - self.pos).length_squared() for p in self.points)

        self.local_points = [p - self.pos for p in points]

    def get_corners(self):
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        return [self.pos + pygame.Vector2(cos_a * lp.x - sin_a * lp.y, sin_a * lp.x + cos_a * lp.y) for lp in self.local_points]

    def update(self, dt):
        self.pos += self.vel * dt
        self.angle += self.angular_velocity * dt

        self.vel *= AIR_RESISTANCE
        self.angular_velocity *= ANGULAR_RESISTANCE
        self.angular_velocity = max(-MAX_ANGULAR_VELOCITY, min(MAX_ANGULAR_VELOCITY, self.angular_velocity))

        MAX_VEL = 300
        if self.vel.length() > MAX_VEL:
            self.vel.scale_to_length(MAX_VEL)

        # Odbicie od krawędzi z korektą pozycji i tłumieniem
        hw = max(abs(lp.x) for lp in self.local_points)
        hh = max(abs(lp.y) for lp in self.local_points)

        if self.pos.x - hw < 0:
            self.pos.x = hw
            self.vel.x *= -0.7
        elif self.pos.x + hw > WIDTH:
            self.pos.x = WIDTH - hw
            self.vel.x *= -0.7

        if self.pos.y - hh < 0:
            self.pos.y = hh
            self.vel.y *= -0.7
            self.angular_velocity *= -0.7
        elif self.pos.y + hh > HEIGHT:
            self.pos.y = HEIGHT - hh
            self.vel.y *= -0.7
            self.angular_velocity *= -0.7

    def draw(self, surface):
        points = self.get_corners()
        pygame.draw.polygon(surface, (0, 255, 0), [(p.x, p.y) for p in points])
        if show_triangulation:
            tris = ear_clip_triangulation(points)
            for tri in tris:
                pygame.draw.polygon(surface, (255, 0, 0), [(p.x, p.y) for p in tri], 1)


def project_polygon(axis, points):
    dots = [p.dot(axis) for p in points]
    return min(dots), max(dots)

def overlap_intervals(a_min, a_max, b_min, b_max):
    return min(a_max, b_max) - max(a_min, b_min)

def sat_collision(poly1, poly2):
    corners1 = poly1.get_corners()
    corners2 = poly2.get_corners()

    axes = []
    n1 = len(corners1)
    n2 = len(corners2)

    for i in range(n1):
        edge = corners1[(i + 1) % n1] - corners1[i]
        axes.append(pygame.Vector2(-edge.y, edge.x).normalize())
    for i in range(n2):
        edge = corners2[(i + 1) % n2] - corners2[i]
        axes.append(pygame.Vector2(-edge.y, edge.x).normalize())

    min_overlap = float('inf')
    mtv_axis = None
    for axis in axes:
        min1, max1 = project_polygon(axis, corners1)
        min2, max2 = project_polygon(axis, corners2)
        overlap = overlap_intervals(min1, max1, min2, max2)
        if overlap <= 0:
            return False, None, None
        if overlap < min_overlap:
            min_overlap = overlap
            mtv_axis = axis

    return True, min_overlap, mtv_axis

def resolve_polygon_collision(a, b):
    collided, overlap, axis = sat_collision(a, b)
    if not collided:
        return

    direction = (b.pos - a.pos)
    if direction.length_squared() == 0:
        direction = pygame.Vector2(1, 0)
    else:
        direction = direction.normalize()
    if axis.dot(direction) < 0:
        axis = -axis

    normal = axis
    tangent = pygame.Vector2(-normal.y, normal.x)

    # --- Znajdź punkt kontaktu (przykład: najbliższe punkty wielokątów)
    corners_a = a.get_corners()
    corners_b = b.get_corners()

    # Znajdź najbliższą parę punktów
    min_dist = float('inf')
    contact_point = None
    for pa in corners_a:
        for pb in corners_b:
            dist = (pa - pb).length_squared()
            if dist < min_dist:
                min_dist = dist
                contact_point = (pa + pb) / 2

    a.pos -= axis * (overlap / 2)
    b.pos += axis * (overlap / 2)

    relative_vel = b.vel - a.vel
    vel_along_normal = relative_vel.dot(normal)
    if vel_along_normal > 0:
        return

    e = 0.7
    j = -(1 + e) * vel_along_normal
    j /= (1 / a.mass + 1 / b.mass)

    impulse = j * normal
    a.vel -= impulse / a.mass
    b.vel += impulse / b.mass

    tangent_vel = relative_vel.dot(tangent)
    jt = -tangent_vel
    jt /= (1 / a.mass + 1 / b.mass)
    jt = max(-j * FRICTION_COEFF, min(j * FRICTION_COEFF, jt))

    friction_impulse = jt * tangent
    a.vel -= friction_impulse / a.mass
    b.vel += friction_impulse / b.mass

    percent = 1
    slop = 0.02
    correction_mag = max(overlap - slop, 0) / (1 / a.mass + 1 / b.mass) * percent
    correction = correction_mag * normal
    a.pos -= correction / a.mass
    b.pos += correction / b.mass

    ra = contact_point - a.pos
    rb = contact_point - b.pos

    delta_angular_a = -ra.cross(impulse) / a.inertia
    delta_angular_b = rb.cross(impulse) / b.inertia

    max_delta = 2.0
    delta_angular_a = max(-max_delta, min(max_delta, delta_angular_a))
    delta_angular_b = max(-max_delta, min(max_delta, delta_angular_b))

    a.angular_velocity += delta_angular_a
    b.angular_velocity += delta_angular_b



poly_bodies = []
for _ in range(10):
    cx = random.randint(100, WIDTH - 100)
    cy = random.randint(100, HEIGHT - 100)
    r = random.randint(30, 70)
    v_count = random.randint(5, 8)
    pts = generate_random_polygon(cx, cy, r, v_count)
    poly_bodies.append(PolygonBody(pts, mass=2,
                     vx=random.uniform(-100, 100), vy=random.uniform(-100, 100),
                     angle=random.uniform(0, 2 * math.pi), angular_velocity=random.uniform(-2, 2)))

running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                show_triangulation = not show_triangulation
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.Vector2(event.pos)
            for poly in poly_bodies:
                dir_vec = poly.pos - mouse_pos
                dist = dir_vec.length()
                if dist < REPULSE_RADIUS and dist != 0:
                    force_dir = dir_vec.normalize()
                    force_mag = REPULSE_FORCE * (1 - dist / REPULSE_RADIUS)
                    impulse = force_dir * force_mag / poly.mass
                    poly.vel += impulse

    for poly in poly_bodies:
        poly.update(dt)
        poly.draw(screen)

    for _ in range(5):
        for i in range(len(poly_bodies)):
            for j in range(i + 1, len(poly_bodies)):
                resolve_polygon_collision(poly_bodies[i], poly_bodies[j])

    pygame.display.flip()
show_triangulation = False

pygame.quit()
