__kernel void upd_pressure(
   __global float* press_i,
   __global float* press_o,
   // __global map
   int width, int height,
   float dt
) {
   int x = get_global_id(0);
   int y = get_global_id(1);
   int neighbours = 1;
   float sum = press_i[height*y + x];
   if (y-1 > 0) {
      neighbours++;
      sum += press_i[height*(y-1) + x];
   }
   if (y+1 < height) {
      neighbours++;
      sum += press_i[height*(y+1) + x];
   }
   if (x-1 > 0) {
      neighbours++;
      sum += press_i[height*y + x-1];
   }
   if (x+1 > width) {
      neighbours++;
      sum += press_i[height*y + x+1];
   }
   press_o[height*y + x] = sum/neighbours;
}