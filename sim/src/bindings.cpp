/**
 * OmniRay pybind11 Bindings
 * ==========================
 * 
 * Exposes the C++ Raycaster to Python as the `raycaster_simd` module.
 * 
 * Python usage:
 *   from raycaster_simd import Raycaster, RaycastResult
 *   
 *   rc = Raycaster(360, 30.0)
 *   rc.add_wall(0, 0, 100, 0)
 *   result = rc.scan(50, 50, 0.0)
 *   print(result.distances)      # list[float]
 *   print(result.query_time_ms)  # float
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "raycaster.h"

namespace py = pybind11;

PYBIND11_MODULE(raycaster_simd, m) {
    m.doc() = "OmniRay AVX2 SIMD Raycaster — 8-way parallel ray-line intersection";

    py::class_<RaycastResult>(m, "RaycastResult",
        "Result of a LiDAR scan: distances + timing")
        .def_readonly("distances", &RaycastResult::distances,
            "List of float distances for each ray")
        .def_readonly("query_time_ms", &RaycastResult::query_time_ms,
            "Time taken for this scan in milliseconds")
        .def("__repr__", [](const RaycastResult& r) {
            return "<RaycastResult rays=" + std::to_string(r.distances.size()) +
                   " time=" + std::to_string(r.query_time_ms) + "ms>";
        });

    py::class_<Raycaster>(m, "Raycaster",
        "AVX2-accelerated 2D raycaster for LiDAR simulation")
        .def(py::init<int, float>(),
            py::arg("num_rays") = 360,
            py::arg("max_range") = 30.0f,
            "Create a raycaster with given ray count and max range")
        .def("add_wall", &Raycaster::add_wall,
            py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
            "Add a wall segment from (x1,y1) to (x2,y2)")
        .def("clear_walls", &Raycaster::clear_walls,
            "Remove all wall segments")
        .def("wall_count", &Raycaster::wall_count,
            "Get current number of walls")
        .def("scan", &Raycaster::scan,
            py::arg("robot_x"), py::arg("robot_y"), py::arg("robot_angle") = 0.0f,
            "Perform full LiDAR scan, returns RaycastResult");
}
