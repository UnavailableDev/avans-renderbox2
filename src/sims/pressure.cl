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

inline int index(int x, int y, int width) {
   return y * width + x;
}

inline float2 bilinearSample(__global const float2* field, float2 pos, int width, int height) {
   int x0 = clamp((int)floor(pos.x), 0, width - 1);
   int x1 = clamp(x0 + 1, 0, width - 1);
   int y0 = clamp((int)floor(pos.y), 0, height - 1);
   int y1 = clamp(y0 + 1, 0, height - 1);

   float sx = pos.x - (float)x0;
   float sy = pos.y - (float)y0;

   float2 v00 = field[index(x0, y0, width)];
   float2 v10 = field[index(x1, y0, width)];
   float2 v01 = field[index(x0, y1, width)];
   float2 v11 = field[index(x1, y1, width)];

   float2 vx0 = mix(v00, v10, sx);
   float2 vx1 = mix(v01, v11, sx);
   return mix(vx0, vx1, sy);
}

inline float2 traceBack(__global const float2* velocityField, float2 pos, float dt, float cellSize, int width, int height) {
   float2 velocity = bilinearSample(velocityField, pos, width, height);
   return pos - (dt / cellSize) * velocity;
}

__kernel void pressure2(
   __global const float* press_i,
   __global float* press_o,
   __global const float2* velocityField,
   int width, int height,
   float dt
) {
   int x = get_global_id(0);
   int y = get_global_id(1);
   
   press_o[width*y + x] = velocityField[width*y + x].x;
}

__kernel void advectVelocity(
   __global const float2* velocityField,
   __global float2* outVelocity,
   int width,
   int height,
   float dt,
   float cellSize)
{
   int x = get_global_id(0);
   int y = get_global_id(1);
   if (x >= width || y >= height) return;

   float2 pos = (float2)(x, y);
   float2 prevPos = traceBack(velocityField, pos, dt, cellSize, width, height);
   prevPos.x = clamp(prevPos.x, 0.0f, (float)(width - 1));
   prevPos.y = clamp(prevPos.y, 0.0f, (float)(height - 1));

   float2 newVelocity = bilinearSample(velocityField, prevPos, width, height);
   outVelocity[index(x, y, width)] = newVelocity;
}