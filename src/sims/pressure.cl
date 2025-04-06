__kernel void upd_pressure(
   __global float* press,
   // __global map
   int width, int height,
   float dt
) {
   int x = get_global_id(0);
   int y = get_global_id(1);
}