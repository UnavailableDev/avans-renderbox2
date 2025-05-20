# eulerian fluid simulation using OpenCL

## TODO
- [x] POC => Pressure dispersion (4h)
   - [x] Debug (1h)
- [ ] Sink/ [ ]Source/ [x]Object (1h)
- [x] file loading (3h)
- [?] speed optimization (0,5h)
- [ ] Velocities (4h)
   - [ ] update(gravity)
   - [ ] Divergence (incompressibility)
   - [ ] Advection
   - [ ] Overrelexation
- [ ] Updated pressure math
- [x] png loading (1h)
- [ ] config/debugging (3h)

pressure seems to be inverted when simulating with wing
velocities are possibly not being generated at the x=0
random vel init causes neat solution with 0 vel init doing nothing at all