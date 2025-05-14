float get_flow_rate(int x, int y, int width, int height, __global float* map) {
   // Check if the coordinates are within the bounds of the map
   // and return 0.0f if they are not
   if (x < 0 || x >= width || y < 0 || y >= height) {
      return 0.0f;
   }
   // Calculate the index in the map array
   int index = y * width + x;
   return map[index];

   // return 1.0f;
}


__kernel void upd_pressure(
   __global const float* press_i,
   __global float* press_o,
   __global float* map,
   int width, int height,
   float dt
) {
   int x = get_global_id(0);
   int y = get_global_id(1);
   int neighbours = 0;
   float sum = 0.0f;

   #define FLOWABLE(XX, YY) if (get_flow_rate(XX, YY, width, height, map) > 0.5f) {neighbours++; sum += press_i[width*(YY) + XX];}
   FLOWABLE(x, y);
   if (neighbours == 0) {
      press_o[width*y + x] = 0.0f;
      return;
   } 
   FLOWABLE(x, y-1);
   FLOWABLE(x, y+1);
   FLOWABLE(x-1, y);
   FLOWABLE(x+1, y);

   /** Average pressure */
   press_o[width*y + x] = sum/neighbours;
}