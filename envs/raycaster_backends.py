"""
OmniRay Raycasting Backends
============================

Unified interface for raycasting with three backends:
  - 'numpy'  : NumPy vectorized (default, no build needed)
  - 'pymunk' : PyMunk segment_query (physics-backed)
  - 'simd'   : C++ AVX2 SIMD via pybind11 (fastest, requires build)

Each backend implements the same Raycaster interface:
    raycaster = create_raycaster(backend='numpy', max_range=30.0, num_rays=360)
    raycaster.set_walls(walls)                # [(x1,y1,x2,y2), ...]
    distances = raycaster.scan(x, y, theta)   # → np.ndarray of shape (num_rays,)
"""

import numpy as np
from abc import ABC, abstractmethod


class RaycasterBase(ABC):
    """Abstract base for all raycasting backends."""

    def __init__(self, num_rays: int = 360, max_range: float = 30.0):
        self.num_rays = num_rays
        self.max_range = max_range

    @abstractmethod
    def set_walls(self, walls: list):
        """Set wall segments [(x1, y1, x2, y2), ...]"""
        ...

    @abstractmethod
    def scan(self, robot_x: float, robot_y: float, robot_angle: float = 0.0) -> np.ndarray:
        """Cast rays and return distances array of shape (num_rays,)."""
        ...


# ---------------------------------------------------------------------------
# Backend: NumPy Vectorized
# ---------------------------------------------------------------------------

class NumpyRaycaster(RaycasterBase):
    """
    Fully vectorized ray-segment intersection using NumPy broadcasting.
    Traces ALL rays × ALL walls in a single batched operation.
    No external dependencies beyond NumPy.
    """

    def __init__(self, num_rays: int = 360, max_range: float = 30.0):
        super().__init__(num_rays, max_range)
        self._walls = None
        self._wall_sx = None
        self._wall_sy = None

    def set_walls(self, walls: list):
        self._walls = np.array(walls, dtype=np.float64)
        self._wall_sx = self._walls[:, 2] - self._walls[:, 0]
        self._wall_sy = self._walls[:, 3] - self._walls[:, 1]

    def scan(self, robot_x: float, robot_y: float, robot_angle: float = 0.0) -> np.ndarray:
        if self._walls is None:
            return np.full(self.num_rays, self.max_range, dtype=np.float32)

        angles = np.linspace(
            robot_angle,
            robot_angle + 2 * np.pi,
            self.num_rays,
            endpoint=False,
            dtype=np.float64,
        )
        dx = np.cos(angles)  # (R,)
        dy = np.sin(angles)  # (R,)

        sx = self._wall_sx  # (W,)
        sy = self._wall_sy  # (W,)

        # Broadcasting: (R, 1) vs (1, W) → (R, W)
        denom = dx[:, None] * sy[None, :] - dy[:, None] * sx[None, :]

        diffx = self._walls[:, 0][None, :] - robot_x
        diffy = self._walls[:, 1][None, :] - robot_y

        safe_denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

        t = (diffx * sy[None, :] - diffy * sx[None, :]) / safe_denom
        u = (diffx * dy[:, None] - diffy * dx[:, None]) / safe_denom

        valid = (
            (t >= 0) & (t <= self.max_range) &
            (u >= 0) & (u <= 1) &
            (np.abs(denom) >= 1e-12)
        )
        t_valid = np.where(valid, t, self.max_range)
        return np.min(t_valid, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Backend: PyMunk Segment Query
# ---------------------------------------------------------------------------

class PymunkRaycaster(RaycasterBase):
    """
    Raycasting via PyMunk's C-backed segment_query_first.
    Good for environments that already use PyMunk for physics.
    """

    def __init__(self, num_rays: int = 360, max_range: float = 30.0):
        super().__init__(num_rays, max_range)
        import pymunk
        self._pymunk = pymunk
        self._space = pymunk.Space()
        self._filter = None

    def set_walls(self, walls: list):
        pymunk = self._pymunk
        # Clear existing shapes
        for shape in list(self._space.shapes):
            self._space.remove(shape)
        for body in list(self._space.bodies):
            self._space.remove(body)

        for x1, y1, x2, y2 in walls:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            seg = pymunk.Segment(body, (x1, y1), (x2, y2), 1.0)
            seg.elasticity = 0.0
            self._space.add(body, seg)

    def scan(self, robot_x: float, robot_y: float, robot_angle: float = 0.0) -> np.ndarray:
        distances = np.full(self.num_rays, self.max_range, dtype=np.float32)
        for i in range(self.num_rays):
            angle = robot_angle + i * 2 * np.pi / self.num_rays
            end_x = robot_x + self.max_range * np.cos(angle)
            end_y = robot_y + self.max_range * np.sin(angle)
            query = self._space.segment_query_first(
                (robot_x, robot_y), (end_x, end_y), 0.0, self._pymunk.ShapeFilter()
            )
            if query is not None and query.shape is not None:
                distances[i] = query.alpha * self.max_range
        return distances


# ---------------------------------------------------------------------------
# Backend: C++ AVX2 SIMD (via pybind11)
# ---------------------------------------------------------------------------

class SimdRaycaster(RaycasterBase):
    """
    Wraps the C++ AVX2 SIMD raycaster built with pybind11.
    Must be compiled first (see sim/CMakeLists.txt).
    """

    def __init__(self, num_rays: int = 360, max_range: float = 30.0):
        super().__init__(num_rays, max_range)
        try:
            from sim import raycaster_simd
            self._engine = raycaster_simd.Raycaster(num_rays, max_range)
        except ImportError as e:
            raise ImportError(
                "C++ SIMD raycaster not built. Run:\n"
                "  cd sim && mkdir build && cd build && cmake .. && cmake --build .\n"
                f"Original error: {e}"
            )

    def set_walls(self, walls: list):
        self._engine.clear_walls()
        for x1, y1, x2, y2 in walls:
            self._engine.add_wall(float(x1), float(y1), float(x2), float(y2))

    def scan(self, robot_x: float, robot_y: float, robot_angle: float = 0.0) -> np.ndarray:
        result = self._engine.scan(float(robot_x), float(robot_y), float(robot_angle))
        return np.array(result.distances, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

BACKENDS = {
    "numpy": NumpyRaycaster,
    "pymunk": PymunkRaycaster,
    "simd": SimdRaycaster,
}


def create_raycaster(
    backend: str = "numpy",
    num_rays: int = 360,
    max_range: float = 30.0,
) -> RaycasterBase:
    """
    Factory for raycasting backends.

    Args:
        backend: 'numpy', 'pymunk', or 'simd'
        num_rays: Number of LiDAR rays per scan
        max_range: Maximum ray distance

    Returns:
        RaycasterBase instance
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(BACKENDS.keys())}")
    return BACKENDS[backend](num_rays=num_rays, max_range=max_range)
